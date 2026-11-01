"""
Pre-processing Method: Distributed Reweighing on Spark
Algorithm: Kamiran & Calders (2012)
Assigns sample weights before model training so that the joint probability
of protected status and favorable outcome equals their expected independent probability:
W(s, y) = (P(s) * P(y)) / P(s, y)
"""

from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression

class DistributedReweighing:
    def __init__(self, protected_col="protected", label_col="label", weight_col="reweight"):
        self.protected_col = protected_col
        self.label_col = label_col
        self.weight_col = weight_col
        self.weights = {}

    def compute_weights(self, df):
        """
        Computes the reweighing factor for each (protected, label) combination
        using distributed Spark group counts.
        """
        total = float(df.count())
        counts = df.groupBy(self.protected_col, self.label_col).count().collect()
        
        # N(s, y)
        n_sy = {(int(r[self.protected_col]), int(r[self.label_col])): float(r["count"]) for r in counts}
        
        # Marginal counts: N(s) and N(y)
        n_s = {}
        n_y = {}
        for (s, y), cnt in n_sy.items():
            n_s[s] = n_s.get(s, 0.0) + cnt
            n_y[y] = n_y.get(y, 0.0) + cnt
            
        # Formula: W(s, y) = [P(s) * P(y)] / P(s, y) = [N(s) * N(y)] / [N * N(s, y)]
        self.weights = {}
        for (s, y), cnt in n_sy.items():
            if cnt > 0:
                w = (n_s[s] * n_y[y]) / (total * cnt)
            else:
                w = 1.0
            self.weights[(s, y)] = w
            
        return self.weights

    def transform(self, df):
        """
        Attaches the computed sample weight as a column in the Spark DataFrame
        using an efficient Spark SQL conditional expression (zero shuffle).
        """
        cond = None
        for (s, y), w in self.weights.items():
            clause = (F.col(self.protected_col) == s) & (F.col(self.label_col) == y)
            if cond is None:
                cond = F.when(clause, float(w))
            else:
                cond = cond.when(clause, float(w))
                
        if cond is not None:
            return df.withColumn(self.weight_col, cond.otherwise(1.0))
        return df.withColumn(self.weight_col, F.lit(1.0))

    def fit_model(self, train_df, features_col="features"):
        """
        Calculates reweighing factors, applies them to train_df, and trains
        a weighted Spark MLlib LogisticRegression model.
        """
        self.compute_weights(train_df)
        weighted_train_df = self.transform(train_df)
        
        lr = LogisticRegression(
            featuresCol=features_col,
            labelCol=self.label_col,
            weightCol=self.weight_col,
            maxIter=50,
            regParam=0.01
        )
        model = lr.fit(weighted_train_df)
        return model, weighted_train_df
