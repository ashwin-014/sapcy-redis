import os
import torch
import redis
import spacy
from spacy.tokens import Doc, Span, Token
from spacy.language import Language
from redis.commands.search import Search
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


# connect to redis
DEFAULT_HOST = os.getenv("REDIS_HOST")
DEFAULT_PORT = os.getenv("REDIS_PORT")
DEFAULT_USER = os.getenv("REDIS_USER")
DEFAULT_PASSWD = os.getenv("REDIS_PASSWD")


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


# query for FT search
def ft_search(redis_conn, user_query):
    q = Query(f"{user_query}").return_fields("headline", "publisher", "label", "score").paging(0, 5)
    docs = redis_conn.ft().search(q).docs

    for doc in docs:
        print("********DOCUMENT: " + str(doc.id) + " ********")
        print(doc.headline)
        print(doc.publisher)

    return docs


# query for similarity
def vec_sim_search(redis_conn, query_vector, **kwargs):
    vector_index = "headline_vector"  # This given as a param throws syntax error
    q = (
        Query(f"*=>[KNN $K @{vector_index} $BLOB]")
        .return_fields("headline", "publisher", "label", "score")
        .sort_by("__headline_vector_score")
        .paging(0, 5)
        .dialect(2)
    )
    #  get K, max_page, vector_index and score_label from kwargs
    params_dict = {"K": 4, "BLOB": query_vector.tobytes()}
    docs = redis_conn.ft().search(q, params_dict).docs
    result = [{"id": doc.id, "text": doc.headline} for doc in docs]
    return result


# Combine vecsim and FT search
def ft_vec_sim_search(redis_conn, user_query, query_vector):
    q = (
        Query(f"(Agilent @label:{{GuruFocus}})=>[KNN $K @headline_vector $BLOB]")
        .return_fields("headline", "publisher", "label", "score")
        .sort_by("__headline_vector_score")
        .paging(0, 5)
        .dialect(2)
    )

    params_dict = {"K": 5, "BLOB": query_vector.tobytes()}
    docs = redis_conn.ft().search(q, params_dict).docs
    return docs


class SimilarityHook:
    """
    User hook which replaces the similarity tag with results from Vector search in Redis
    """

    pass


@Language.factory(
    "sentence_transformer",
    default_config={"model_name": ""},
)
class SentenceTrf:
    models = {}

    def __init__(self, nlp, name, model_name: str) -> None:
        self.model = self.get_model(model_name)
        if not Doc.has_extension("sentence_trf"):
            Doc.set_extension("sentence_trf", default=[])

    def __call__(self, doc):
        vector = self.model.encode([doc.text])[0]
        doc._.sentence_trf = vector
        return doc

    def get_model(self, model_name: str) -> SentenceTransformer:
        model = SentenceTransformer(model_name)
        return model


@Language.factory(
    "qa_transformer",
    default_config={"model_name": ""},
)
class QATrf:
    models = {}

    def __init__(self, nlp, name, model_name: str) -> None:
        self.get_model(model_name)
        if not Doc.has_extension("qa_results"):
            Doc.set_extension("qa_results", default=[])

    def __call__(self, doc):
        for context in doc._.similar_docs:
            inputs = self.tokenizer(
                doc.text,
                context["text"],
                padding="max_length",
                return_tensors="pt",
                truncation="only_second",
            )
            with torch.no_grad():
                outputs = self.model(**inputs)

            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()

            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            doc._.qa_results.append(
                {"id": context["id"], "answer": self.tokenizer.decode(predict_answer_tokens)}
            )
        return doc

    def get_model(self, model_name: str) -> SentenceTransformer:
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


@Language.factory(
    "vector_search",
    default_config={
        "redis_host": "",
        "redis_port": 0,
        "query_mode": "",
        "K": 0,
        "max_page": 0,
        "vector_index": "",
        "score_label": "",
    },
)
def create_vector_search_component(
    nlp,
    name,
    redis_host="",
    redis_port=0,
    query_mode="",
    K=0,
    max_page=0,
    vector_index="",
    score_label="",
):
    kwargs = {
        "redis_host": redis_host,
        "redis_port": redis_port,
        "query_mode": query_mode,
        "K": K,
        "max_page": max_page,
        "vector_index": vector_index,
        "score_label": score_label,
    }
    return VectorSearchComponent(**kwargs)


class VectorSearchComponent:
    def __init__(self, **kwargs):
        if kwargs["redis_host"] and kwargs["redis_port"]:
            self.redis_conn = redis.Redis(host=kwargs["redis_host"], port=kwargs["redis_port"])
        else:
            self.redis_conn = get_connection()

        self.query_mode = kwargs["query_mode"]
        if not Doc.has_extension("similar_docs"):
            Doc.set_extension("similar_docs", default=[])
        self.kwargs = kwargs

    def __call__(self, doc: Doc) -> Doc:
        similar_docs = []
        if self.query_mode == "ft_search":
            similar_docs = ft_search(self.redis_conn, doc.text)

        elif self.query_mode == "vector_search":
            similar_docs = vec_sim_search(self.redis_conn, doc._.sentence_trf, **self.kwargs)

        elif self.query_mode == "ft_vector_search":
            similar_docs = ft_vec_sim_search(self.redis_conn, doc._.sentence_trf, **self.kwargs)

        else:
            raise KeyError("Improper query_mode")

        doc._.similar_docs = similar_docs
        return doc


def main():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(
        "sentence_transformer",
        config={"model_name": "sentence-transformers/all-distilroberta-v1"},
    )
    nlp.add_pipe(
        "vector_search",
        config={"redis_host": "", "redis_port": 0, "query_mode": "vector_search"},
    )
    nlp.add_pipe(
        "qa_transformer",
        config={"model_name": "deepset/roberta-base-squad2"},
        last=True,
    )
    print(nlp.pipe_names)
    doc = nlp("Agilent Awards Trilogy Sciences with a Golden Ticket at LabCentral")
    print(doc._.similar_docs)
    print(doc._.qa_results)

    vec_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
    redis_conn = get_connection()
    vec_sim_search(
        redis_conn,
        vec_model.encode("Agilent Awards Trilogy Sciences with a Golden Ticket at LabCentral"),
    )


if __name__ == "__main__":
    main()
