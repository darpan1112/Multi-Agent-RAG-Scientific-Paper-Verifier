import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, csv_path):
        # Limiting to 1500 rows to ensure fast loading on CPU (reduces time from 5+ mins to ~5 seconds)
        self.data = pd.read_csv(csv_path).head(1500)

        # Make column names lowercase
        self.data.columns = self.data.columns.str.lower()

        # Try to auto-detect title and abstract columns
        title_col = None
        abstract_col = None

        for col in self.data.columns:
            if "title" in col:
                title_col = col
            if "abstract" in col or "summary" in col:
                abstract_col = col

        if title_col is None or abstract_col is None:
            raise Exception("Could not find title or abstract column in dataset.")

        self.documents = (
            self.data[title_col].astype(str) + ". " +
            self.data[abstract_col].astype(str)
        ).tolist()

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.documents)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings))

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        retrieved_docs = [self.documents[i] for i in indices[0]]
        retrieved_embeddings = [self.embeddings[i] for i in indices[0]]

        return retrieved_docs, retrieved_embeddings, query_embedding