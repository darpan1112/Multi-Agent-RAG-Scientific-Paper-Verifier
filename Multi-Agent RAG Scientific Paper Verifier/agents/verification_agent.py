import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VerificationAgent:
    def verify(self, query_embedding, retrieved_embeddings):
        similarities = cosine_similarity(query_embedding, retrieved_embeddings)[0]

        avg_similarity = np.mean(similarities)
        evidence_density = np.sum(similarities > 0.5) / len(similarities)

        final_score = (avg_similarity * 0.7) + (evidence_density * 0.3)

        if final_score > 0.65:
            verdict = "Supported"
        elif final_score > 0.4:
            verdict = "Partially Supported"
        else:
            verdict = "Not Supported"

        return verdict, final_score, similarities