# Databricks notebook source


# COMMAND ----------

# MAGIC %pip install -e ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys
from pathlib import Path

import yaml
from loguru import logger
import mlflow
from pyspark.sql import SparkSession
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(str(Path.cwd().parent / "src"))

from marvelous.logging import setup_logging
from marvelous.common import is_databricks
from us_accidents.config import ProjectConfig, Tags
# from us_accidents.models.basic_model import BasicModel

# Load configuration
config_path = os.path.abspath(os.path.join(Path.cwd(), "..", "project_config.yaml"))
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

setup_logging(log_file="logs/marvelous-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


# COMMAND ----------

# Generate two DFs from our training and testing datasets

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

catalog_name = "mlops_dev" # need to parameterize this
schema_name = "corretco"
num_features = config.num_features
cat_features = config.cat_features
target = config.target

def load_data() -> None:
    """Load training and testing data from Delta tables.

    Splits data into features (X_train, X_test) and target (y_train, y_test).
    """
    logger.info("🔄 Loading data from Databricks tables...")
    train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
    test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
    data_version = "0"  # describe history -> retrieve

    X_train = train_set[num_features + cat_features]
    y_train = train_set[target]
    X_test = test_set[num_features + cat_features]
    y_test = test_set[target]
    logger.info("✅ Data successfully loaded.")
    return X_train, X_test, y_train, y_test

load_data()

# COMMAND ----------

# Work on our model

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train():
    """Encode categorical features and define a preprocessing pipeline.

    Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
    features. Constructs a pipeline combining preprocessing and LightGBM regression model.
    """
    logger.info("🔄 Defining preprocessing pipeline...")
    clf_base = RandomForestClassifier()
    grid = {'n_estimators': [10, 50, 100],
            'max_features': ['sqrt']}
    clf_rf = GridSearchCV(clf_base, grid, cv=5, n_jobs=8, scoring='f1_macro')
    logger.info("✅ Preprocessing pipeline defined.")
    
    logger.info("🚀 Starting training...")
    clf_rf.fit(X_train, y_train)
    logger.info("✅ Completed training...")
    
    return clf_rf

train()
    

# COMMAND ----------

# Set tracking uri, to be able to work locally too
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="prd")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------

# Create experiment: mlflow.set_experiment()
# Add tag to experiment: mlflow.set_experiment_tags() / or add tags: {"yyy": "xxx"} to experiment. Add git commit hash? And more?

# COMMAND ----------

# Start a run, with a WITH statement
# Logg metrics, list of column ... Or perhaps better: mlflow.<model_flavor>.autolog()

# Log the input too, very important: mlflow.log_input()
# When the model is created, also log the signature!
# To log the model: mlflow.<model_flavor>.log_model()

# COMMAND ----------

# Fetch our experiment using a name and tag
# Fund the latest run_id (we trust it's the best)
# Find the logged model

# COMMAND ----------

# Register model
# Two ways. With mlflow.register_model(), we can pass tags
# And it seems
