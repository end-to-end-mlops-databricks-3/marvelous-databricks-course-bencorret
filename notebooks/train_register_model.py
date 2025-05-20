# Databricks notebook source


# COMMAND ----------

# MAGIC %pip install -e ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession

# Add the src directory to the Python path
sys.path.append(str(Path.cwd().parent / "src"))

from marvelous.common import is_databricks
from marvelous.logging import setup_logging

from us_accidents.config import ProjectConfig, Tags
from us_accidents.models.model import BasicModel

# Load configuration
config_path = os.path.abspath(os.path.join(Path.cwd(), "..", "project_config.yaml"))
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

setup_logging(log_file="logs/marvelous-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config_path = os.path.abspath(os.path.join(Path.cwd(), "..", "project_config.yaml"))
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2", "job_run_id": "job-12345"})

# COMMAND ----------
# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
basic_model.load_data()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
basic_model.train()
basic_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/us-accidents-basic"], filter_string="tags.branch='week2'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/random-forest-classifier-model")

# COMMAND ----------
# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = basic_model.load_latest_model_and_predict(X_test)
# COMMAND ----------
