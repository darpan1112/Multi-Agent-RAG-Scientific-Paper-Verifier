class ExplanationAgent:
    def generate(self, query, docs, similarities, verdict, score, uncertainty_level):
        max_sim = max(similarities)
        
        explanation = f"""
**Claim Assessed:** {query}
**Final Verdict:** {verdict}

**🔍 How to read these metrics:**
* **Confidence Score ({round(score, 2)}):** Indicates how certain the system is about the final verdict. A higher score means stronger supporting evidence was found.
* **Uncertainty Level ({uncertainty_level}):** Measures consistency across the top evidence. 'Low Uncertainty' means the retrieved papers agree with each other. 'High Uncertainty' means there are contradictory opinions or diverse claims among the papers.
* **Top Similarity Score ({round(max_sim, 2)}):** Represents how closely your claim matches the single most relevant scientific paper in the database (1.0 = exact 100% text match).

*Reasoning: The decision is based on semantic understanding of the claim and calculating text similarity across multiple retrieved research papers.*
"""

        return explanation