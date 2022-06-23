import os
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.field import TagField

# from redis.commands.search.query import Query
# from redis.commands.search.result import Result
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_columns", 10)

vec_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

# constants
DEFAULT_HOST = os.getenv("REDIS_HOST")
DEFAULT_PORT = os.getenv("REDIS_PORT")
DEFAULT_USER = os.getenv("REDIS_USER")
DEFAULT_PASSWD = os.getenv("REDIS_PASSWD")
# redis-15873.c284.us-east1-2.gce.cloud.redislabs.com:15873
NUMBER_ARTICLES = 300
VECTOR_FIELD_NAME = "headline_vector"
DISTANCE_METRIC = "COSINE"
TEXT_EMBEDDING_DIMENSION = 768
NUMBER_HEADLINES = 300
# ITEM_KEYWORD_EMBEDDING_FIELD='item_keyword_vector'


def get_connection(
    host: str = "", port: int = 0, username: str = "", password: str = ""
) -> redis.Redis:
    return redis.StrictRedis(
        host=host if host else DEFAULT_HOST,
        port=port if port else DEFAULT_PORT,
        username=username if username else DEFAULT_USER,
        password=password if password else DEFAULT_PASSWD,
        decode_responses=True,
    )


# Generate embeddings (vectors) for each headline
def get_vectors(text):
    # headline_vectors = [vec_model.encode(sentence) for sentence in result_df["headline"]]
    return vec_model.encode(text)
    # check how many dimensions in a single vector
    # print(headline_vectors[0].shape)
    # return headline_vectors


def load_vectors(client: redis.StrictRedis, headline_df, vector_field_name):
    p = client.pipeline(transaction=False)
    for index, row in headline_df.iterrows():
        # hash key
        key = "headline:" + str(index) + ":" + str(row["ID"])

        # hash values
        row[vector_field_name] = row["vector"].astype(np.float32).tobytes()

        item_metadata = row.to_dict()
        del item_metadata["vector"]

        # HSET
        p.hset(key, mapping=item_metadata)

    p.execute()


def create_flat_index(
    redis_conn, vector_field_name, number_of_vectors, vector_dimensions=512, distance_metric="L2"
):
    redis_conn.ft().create_index(
        [
            VectorField(
                vector_field_name,
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": vector_dimensions,
                    "DISTANCE_METRIC": distance_metric,
                    "INITIAL_CAP": number_of_vectors,
                    "BLOCK_SIZE": number_of_vectors,
                },
            ),
            TextField("stock"),
            # headline,url,publisher,date,stock
            TextField("headline"),
            TextField("url"),
            TextField("publisher"),
        ]
    )


def create_hnsw_index(
    redis_conn,
    vector_field_name,
    number_of_vectors,
    vector_dimensions=512,
    distance_metric="L2",
    M=40,
    EF=200,
):
    redis_conn.ft().create_index(
        [
            VectorField(
                vector_field_name,
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": vector_dimensions,
                    "DISTANCE_METRIC": distance_metric,
                    "INITIAL_CAP": number_of_vectors,
                    "M": M,
                    "EF_CONSTRUCTION": EF,
                },
            ),
            TextField("stock"),
            # headline,url,publisher,date,stock
            TextField("headline"),
            TextField("url"),
            TextField("publisher"),
        ]
    )


def main():
    headlines_df = pd.read_csv("./300_stock_headlines.csv")
    # headlines_df.drop("Unnamed: 0.1", axis=1, inplace=True)
    headlines_df = headlines_df.rename(columns={"Unnamed: 0": "ID"})
    # print(headlines_df.head())
    headlines_df.drop("Unnamed: 0.1", axis=1, inplace=True)
    headlines_df.reset_index()
    headlines_df.head(5)

    result_df = headlines_df.copy()
    result_df["vector"] = result_df["headline"].apply(lambda x: get_vectors(x))

    redis_conn = get_connection()
    redis_conn.flushall()
    create_flat_index(redis_conn, VECTOR_FIELD_NAME,NUMBER_HEADLINES,TEXT_EMBEDDING_DIMENSION,'COSINE')
    load_vectors(redis_conn, result_df, VECTOR_FIELD_NAME)
    print("Articles loaded and flat indexed")

    redis_conn.flushall()
    create_hnsw_index(redis_conn, VECTOR_FIELD_NAME,NUMBER_HEADLINES,TEXT_EMBEDDING_DIMENSION,'COSINE',M=40,EF=200)
    load_vectors(redis_conn, result_df, VECTOR_FIELD_NAME)

    print("Articles loaded and hnsw indexed")


if __name__ == "__main__":
    conn = get_connection()
    print(conn)
    print ('Loading and Indexing + ' +  str(NUMBER_HEADLINES) + ' products')

    #flush all data
    conn.flushall()
    main()
