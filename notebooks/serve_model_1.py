# Databricks notebook source


# COMMAND ----------

# MAGIC %pip install -e ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import time
import sys
from pathlib import Path

import mlflow
import requests
import yaml
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# Add the src directory to the Python path
sys.path.append(str(Path.cwd().parent / "src"))

from us_accidents.config import ProjectConfig, Tags
from us_accidents.serving.model_serving import ModelServing

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

# Create a sample request body
required_columns = [
    "Timezone_US/Eastern",
    "Timezone_US/Mountain",
    "Timezone_US/Pacific",
    "Weekday_1",
    "Weekday_2",
    "Weekday_3",
    "Weekday_4",
    "Weekday_5",
    "Weekday_6",
    "Station",
    "Stop",
    "Traffic_Signal",
    "Severity_4",
    "Rd",
    "St",
    "Dr",
    "Ave",
    "Blvd",
    "I-",
    "Astronomical_Twilight_Night",
    "Start_Lat",
    "Start_Lng",
    "Pressure_bc"
]

# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Convert decimal columns to float64
test_set["Start_Lat"] = test_set["Start_Lat"].astype("float64")
test_set["Start_Lng"] = test_set["Start_Lng"].astype("float64")
test_set["Pressure_bc"] = test_set["Pressure_bc"].astype("float64")

# Replace nan values with None
test_set = test_set.where(pd.notna(test_set), None)

# Sample 100 records from the training set
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'Timezone_US/Eastern': 0,
   'Timezone_US/Mountain': 0,
   'Timezone_US/Pacific': 1,
   'Weekday_1': 0,
   'Weekday_2': 0,
   'Weekday_3': 1,
   'Weekday_4': 0,
   'Weekday_5': 0,
   'Weekday_6': 0,
   'Station': 0,
   'Stop': 0,
   'Traffic_Signal': 0,
   'Severity_4': 0,
   'Rd': 0,
   'St': 0,
   'Dr': 0,
   'Ave': 0,
   'Blvd': 0,
   'I-': 0,
   'Astronomical_Twilight_Night': 0,
   'Start_Lat': 47.97825,
   'Start_Lng': -122.177175,
   'Pressure_bc': 29.52}]
"""

def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# Load test
for i in range(len(dataframe_records)):
    try:
        status_code, response_text = call_endpoint(dataframe_records[i])
        print(f"Record number: {i}")
        print(f"Response Status: {status_code}")
        print(f"Response Text: {response_text}")
        time.sleep(0.2)
    except:
        print("Error")
        print(dataframe_records[i])
        raise
