# Databricks notebook source
from pyspark.sql.functions import col, concat_ws, split, explode, array_distinct, when, lower, lit
import re

# COMMAND ----------

data_file_path = "/Volumes/mlops_dev/corretco/data/US_Accidents_March23_short.csv"

# COMMAND ----------

raw_df = spark.read. \
    format("csv") \
    .option("header", "true") \
    .option("separator", ",") \
    .load(data_file_path)

raw_df.createOrReplaceTempView("raw_accidents")
