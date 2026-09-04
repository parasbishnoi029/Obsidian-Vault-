# Vector Databases

Vector databases store embeddings (numerical representations of text,
images, etc.) and let you search by *similarity* instead of exact match.

## Why they matter for RAG
In a retrieval-augmented generation pipeline, you embed your documents
once, store the vectors, and then at query time embed the user's
question and find the nearest stored vectors. Those nearest chunks
become the context you feed to the LLM.

## Popular options
- **Chroma** - lightweight, easy to run locally, good for prototypes.
- **Pinecone** - managed, scales well, has a generous free tier.
- **Weaviate** - open source, supports hybrid (keyword + vector) search.
- **FAISS** - a library (not a full DB) from Meta, very fast for
  in-memory search, no persistence/server out of the box.

## Distance metrics
Cosine similarity and Euclidean (L2) distance are the two most common.
Cosine is usually preferred for text embeddings since it ignores
magnitude and focuses on direction.

#embeddings #rag #databases
