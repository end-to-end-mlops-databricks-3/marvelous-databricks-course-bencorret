# Databricks notebook source
import sys
import os
from pyspark.sql import SparkSession

# Add the src directory to the Python path
current_working_directory = os.getcwd()
relative_path_to_module = os.path.join('..', 'src')
absolute_path_to_module = os.path.abspath(os.path.join(current_working_directory, relative_path_to_module))
sys.path.append(absolute_path_to_module)

from us_accidents.config import ProjectConfig
from us_accidents.data_processor import DataProcessor

# Load configuration
config_path = os.path.abspath(os.path.join(current_working_directory, '..', 'project_config.yaml'))
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")


# COMMAND ----------

# Define impoprtant variables
catalog_name = config.catalog_name
schema_name = config.schema_name
data_file_path = f"/Volumes/{catalog_name}/{schema_name}/data/US_Accidents_March23_short.csv"

# COMMAND ----------

# Load the raw data
spark = SparkSession.builder \
    .appName("US Accidents Data Preprocessing") \
    .getOrCreate()

raw_df = spark.read. \
    format("csv") \
    .option("header", "true") \
    .option("separator", ",") \
    .load(data_file_path)

# COMMAND ----------

# Initialize DataProcessor and clean the data
data_processor = DataProcessor(spark=spark, config=config, dataframe=raw_df)

data_processor.preprocess()


# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()


# COMMAND ----------
# Save to catalog
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
data_processor.enable_change_data_feed()