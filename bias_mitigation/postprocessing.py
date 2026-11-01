"""
Post-processing Method: Reject Option Classification (ROC) on Spark
Algorithm: Kamiran, Karim, & Zhang (2012)
Exploits model uncertainty around the 0.5 decision boundary:
Within a critical uncertainty band [0.5 - theta, 0.5 + theta]:
- Unprivileged instances (S=0) close to the boundary are flipped to favorable (1)
- Privileged instances (S=1) close to the boundary are flipped to unfavorable (0)
Finds the optimal margin theta on a validation split and applies it to test predictions.
"""

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from bias_mitigation.metrics import evaluate_fairness

class DistributedRejectOptionClassifier:
    def __init__(self, protected_col="protected", label_col="label", candidate_thetas=None):
        self.protected_col = protected_col
        self.label_col = label_col
        self.candidate_thetas = candidate_thetas or [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
        self.best_theta = 0.0

    def extract_prob(self, df, prob_col="probability", output_prob_col="prob_1"):
        """Extracts P(Y=1) scalar from Spark MLlib probability vector column."""
        return df.withColumn(output_prob_col, vector_to_array(F.col(prob_col))[1])

    def apply_threshold(self, df, theta, prob_col="prob_1", pred_col="prediction", out_col="mitigated_pred"):
        """
        Applies Reject Option Classification flip logic on the Spark DataFrame:
        - If S=0 and prob in [0.5 - theta, 0.5): flip to 1.0
        - If S=1 and prob in [0.5, 0.5 + theta]: flip to 0.0
        - Otherwise keep original prediction
        """
        cond = F.when(
            (F.col(self.protected_col) == 0) & 
            (F.col(prob_col) >= (0.5 - theta)) & 
            (F.col(prob_col) < 0.5), 
            1.0
        ).when(
            (F.col(self.protected_col) == 1) & 
            (F.col(prob_col) <= (0.5 + theta)) & 
            (F.col(prob_col) >= 0.5), 
            0.0
        ).otherwise(F.col(pred_col))

        return df.withColumn(out_col, cond)

    def fit(self, val_df, prob_col="probability", pred_col="prediction"):
        """
        Searches candidate margins theta on validation DataFrame
        to select theta* that minimizes Statistical Parity Difference.
        """
        val_with_prob = self.extract_prob(val_df, prob_col=prob_col, output_prob_col="prob_1")
        
        best_score = -float("inf")
        best_metrics = None
        
        for theta in self.candidate_thetas:
            transformed_val = self.apply_threshold(
                val_with_prob, theta=theta, prob_col="prob_1", pred_col=pred_col, out_col="mitigated_pred"
            )
            
            m = evaluate_fairness(
                transformed_val,
                protected_col=self.protected_col,
                label_col=self.label_col,
                prediction_col="mitigated_pred"
            )
            
            # Score: High accuracy and Statistical Parity Difference close to 0
            score = m["accuracy"] - (0.8 * abs(m["statistical_parity_diff"]))
            
            if score > best_score:
                best_score = score
                self.best_theta = theta
                best_metrics = m
                
        return self.best_theta, best_metrics

    def transform(self, test_df, prob_col="probability", pred_col="prediction", out_col="mitigated_pred"):
        """Applies optimal theta* to transform test predictions."""
        test_with_prob = self.extract_prob(test_df, prob_col=prob_col, output_prob_col="prob_1")
        return self.apply_threshold(
            test_with_prob, theta=self.best_theta, prob_col="prob_1", pred_col=pred_col, out_col=out_col
        )
