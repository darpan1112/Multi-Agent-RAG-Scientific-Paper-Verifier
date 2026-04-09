import numpy as np

class UncertaintyAgent:
    def calculate_uncertainty(self, similarities):
        variance = np.var(similarities)

        if variance < 0.01:
            level = "Low Uncertainty"
        elif variance < 0.05:
            level = "Moderate Uncertainty"
        else:
            level = "High Uncertainty"

        return variance, level