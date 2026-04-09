from vector_store import VectorStore
from agents.retrieval_agent import RetrievalAgent
from agents.verification_agent import VerificationAgent
from agents.explanation_agent import ExplanationAgent
from agents.uncertainty_agent import UncertaintyAgent

def main():
    print("Initializing Multi-Agent RAG System...")
    
    vector_store = VectorStore("data.csv")

    retrieval_agent = RetrievalAgent(vector_store)
    verification_agent = VerificationAgent()
    explanation_agent = ExplanationAgent()
    uncertainty_agent = UncertaintyAgent()

    query = input("\nEnter scientific claim to verify: ")

    # Retrieval
    docs, embeddings, query_embedding = retrieval_agent.retrieve(query)

    # Verification
    verdict, score, similarities = verification_agent.verify(query_embedding, embeddings)

    # Uncertainty
    variance, uncertainty_level = uncertainty_agent.calculate_uncertainty(similarities)

    # Explanation
    explanation = explanation_agent.generate(
        query, docs, similarities, verdict, score, uncertainty_level
    )

    print(explanation)

if __name__ == "__main__":
    main()