"""
In-processing Method: Fair Grid Search on Spark
Algorithm: Cost-Sensitive Reductions for Fair Classification (Agarwal et al., 2018)
Modifies the model's loss objective during training by adding a fairness penalty lambda.
It searches over penalty parameters lambda in [0, 1] and trains Spark LogisticRegression
with fairness-adjusted cost weights, selecting the model that optimizes the fairness-accuracy balance.
"""

from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from bias_mitigation.metrics import evaluate_fairness

class DistributedFairInProcessing:
    def __init__(self, protected_col="protected", label_col="label", candidate_lambdas=None):
        self.protected_col = protected_col
        self.label_col = label_col
        # Fairness penalty grid
        self.candidate_lambdas = candidate_lambdas or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.best_lambda = 0.0
        self.best_model = None

    def _create_fairness_weights(self, df, lam, weight_col="fair_weight"):
        """
        Adjusts instance costs based on fairness penalty lambda:
        - Penalizes under-predicting favorable outcomes for unprivileged group (S=0, Y=1)
        - Penalizes over-predicting favorable outcomes for privileged group (S=1, Y=0)
        """
        # When lambda = 0, all weights are 1.0 (standard empirical risk minimization)
        # As lambda increases, mistakes on disadvantaged group carry higher cost
        w_unpriv_pos = 1.0 + (1.5 * lam)
        w_unpriv_neg = max(0.2, 1.0 - (0.5 * lam))
        w_priv_pos = max(0.2, 1.0 - (0.5 * lam))
        w_priv_neg = 1.0 + (1.0 * lam)

        cond = F.when((F.col(self.protected_col) == 0) & (F.col(self.label_col) == 1), float(w_unpriv_pos)) \
                .when((F.col(self.protected_col) == 0) & (F.col(self.label_col) == 0), float(w_unpriv_neg)) \
                .when((F.col(self.protected_col) == 1) & (F.col(self.label_col) == 1), float(w_priv_pos)) \
                .otherwise(float(w_priv_neg))

        return df.withColumn(weight_col, cond)

    def fit(self, train_df, val_df, features_col="features"):
        """
        Trains a model for each candidate fairness penalty lambda,
        evaluates on val_df, and selects the model with the best fairness-accuracy trade-off.
        """
        best_score = -float("inf")
        best_metrics = None
        
        for lam in self.candidate_lambdas:
            weighted_df = self._create_fairness_weights(train_df, lam, "fair_weight")
            
            lr = LogisticRegression(
                featuresCol=features_col,
                labelCol=self.label_col,
                weightCol="fair_weight",
                maxIter=30,
                regParam=0.01
            )
            model = lr.fit(weighted_df)
            
            # Evaluate on validation split
            preds = model.transform(val_df)
            m = evaluate_fairness(preds, self.protected_col, self.label_col, "prediction")
            
            # Score: High accuracy while keeping Statistical Parity Difference close to 0
            # score = Accuracy - 1.0 * |SPD|
            spd_penalty = abs(m["statistical_parity_diff"])
            score = m["accuracy"] - (0.8 * spd_penalty)
            
            if score > best_score:
                best_score = score
                self.best_lambda = lam
                self.best_model = model
                best_metrics = m
                
        return self.best_model, self.best_lambda, best_metrics
