"""
Spark Session Helper
Configures Java environment, memory, and parallel workers for PySpark.
"""

import os
import sys

def get_spark_session(app_name="DistributedBiasMitigation", memory="4g", cores="*"):
    """
    Initializes a local SparkSession configured for high-performance tabular ML.
    """
    # Use OpenJDK 21 (required by PySpark 4.x)
    if "JAVA_HOME" not in os.environ or "java-21" not in os.environ["JAVA_HOME"]:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
        
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(f"local[{cores}]") \
        .config("spark.driver.memory", memory) \
        .config("spark.executor.memory", memory) \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "8") \
        .config("spark.ui.enabled", "false") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    return spark
