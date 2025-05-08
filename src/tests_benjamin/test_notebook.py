# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------
schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("city", StringType(), True),
        StructField("country", StringType(), True),
    ]
)

# COMMAND ----------
data = [
    ("1", "Alice", 30, "New York", "USA"),
    ("2", "Bob", 25, "Los Angeles", "USA"),
    ("3", "Charlie", 35, "London", "UK"),
    ("4", "David", 28, "Paris", "France"),
]
df = spark.createDataFrame(data, schema)
# COMMAND ----------
df.show()