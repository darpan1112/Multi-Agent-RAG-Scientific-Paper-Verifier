class RetrievalAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5):
        docs, embeddings, query_embedding = self.vector_store.search(query, top_k)
        return docs, embeddings, query_embedding