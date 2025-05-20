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

# Add the src directory to the Python path
sys.path.append(str(Path.cwd().parent / "src"))

from marvelous.logging import setup_logging

from us_accidents.config import ProjectConfig

# from us_accidents.models.model import xxx

# Load configuration
config_path = os.path.abspath(os.path.join(Path.cwd(), "..", "project_config.yaml"))
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

setup_logging(log_file="logs/marvelous-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


# COMMAND ----------

# Generate two DFs from our training and testing datasets

# COMMAND ----------

# Set tracking uri, to be able to work locally too

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
