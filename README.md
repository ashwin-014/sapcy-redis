# spacy-redis
Integrate Spacy features with the Redis storage engine to scale applications

## Goal
Convert SpaCy into the go-to library for doing NLP using the Redis ecosystem

### WIP
* RedisAI to host Spacy Transformer Models
* Redis Vector Storage for semantic search

#### ToDo
Create SpaCy components for:
* RediSearch for BM25 search
* Combine vector search and BM25
* Transformer model to find the span of searches
* Add support for training the models via SpaCy