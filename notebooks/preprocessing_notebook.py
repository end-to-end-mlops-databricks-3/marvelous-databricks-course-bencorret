# Databricks notebook source

# COMMAND ----------
%pip install -e ..

# COMMAND ----------
%restart_python

# COMMAND ----------

import sys
import os
import yaml
from pathlib import Path
from pyspark.sql import SparkSession
from loguru import logger
# from marvelous.logging import setup_logging
# from marvelous.timer import Timer

# Add the src directory to the Python path
sys.path.append(str(Path.cwd().parent / 'src'))

from us_accidents.config import ProjectConfig
from us_accidents.data_processor import DataProcessor

# Load configuration
config_path = os.path.abspath(os.path.join(Path.cwd(), '..', 'project_config.yaml'))
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

# setup_logging(log_file="logs/marvelous-1.log")
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


# COMMAND ----------

# Define impoprtant variables
catalog_name = config.catalog_name
schema_name = config.schema_name
data_file_path = f"/Volumes/{catalog_name}/{schema_name}/data/US_Accidents_March23_short.csv"

# COMMAND ----------

# Load the raw data
logger.info(f"Building Spark Session")
spark = SparkSession.builder \
    .appName("US Accidents Data Preprocessing") \
    .getOrCreate()

logger.info(f"Loading raw dataframe")
raw_df = spark.read. \
    format("csv") \
    .option("header", "true") \
    .option("separator", ",") \
    .load(data_file_path)

# COMMAND ----------

# Initialize DataProcessor and clean the data
logger.info(f"Initialize DataProcessor object")
data_processor = DataProcessor(spark=spark, config=config, dataframe=raw_df)

logger.info(f"Preprocess data + save table to UC")
data_processor.preprocess()


# COMMAND ----------

# Split the data
logger.info(f"Split data")
X_train, X_test = data_processor.split_data()


# COMMAND ----------
# Save to catalog
logger.info(f"Save train and test sets to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info(f"Enable CDF")
data_processor.enable_change_data_feed()