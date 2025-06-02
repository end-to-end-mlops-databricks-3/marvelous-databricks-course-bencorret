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
from pyspark.dbutils import DBUtils

# Add the src directory to the Python path
sys.path.append(str(Path.cwd().parent / "src"))

from us_accidents.config import ProjectConfig, Tags
from us_accidents.models.serving import ModelServing

# spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load configuration
config_path = os.path.abspath(os.path.join(Path.cwd(), "..", "project_config.yaml"))
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = config.model_names["model1"]
endpoint_name = config.endpoint_names["endpoint1"]

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


# COMMAND ----------
logger.info("Initializing ModelServing class ...")
model_serving = ModelServing(
    model_name = f"{catalog_name}.{schema_name}.{model_name}", endpoint_name = endpoint_name
)

# COMMAND ----------
logger.info("Deploy the model serving endpoint ...")
model_serving.deploy_or_update_serving_endpoint()
logger.info("Completed the deployment of the model serving endpoint.")

# COMMAND ----------
