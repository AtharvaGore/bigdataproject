"""
Spark Feature Pipeline Helper
Automatically indexes categorical features and assembles numerical features into MLlib Vector.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

def build_feature_pipeline(df, feature_cols, categorical_cols):
    """
    Constructs and fits a Spark ML feature pipeline that:
    1. Indexes categorical string columns
    2. One-hot encodes indexed categorical columns
    3. Assembles all features into a single 'features' vector
    """
    stages = []
    encoded_cols = []
    
    for c in categorical_cols:
        indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
        stages.extend([indexer, encoder])
        encoded_cols.append(f"{c}_vec")
        
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    assembler = VectorAssembler(inputCols=numeric_cols + encoded_cols, outputCol="features", handleInvalid="skip")
    stages.append(assembler)
    
    pipeline = Pipeline(stages=stages)
    fitted_pipeline = pipeline.fit(df)
    return fitted_pipeline
