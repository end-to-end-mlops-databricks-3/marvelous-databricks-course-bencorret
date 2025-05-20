# Databricks notebook source

# COMMAND ----------
# %pip install -e ..

# COMMAND ----------
# %restart_python

# COMMAND ----------

import os
import sys
from pathlib import Path

import yaml
from loguru import logger
from pyspark.sql import SparkSession

# Add the src directory to the Python path
sys.path.append(str(Path.cwd().parent / "src"))

from marvelous.logging import setup_logging
from marvelous.timer import Timer

from us_accidents.config import ProjectConfig
from us_accidents.data_processor import DataProcessor

# Load configuration
config_path = os.path.abspath(os.path.join(Path.cwd(), "..", "project_config.yaml"))
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

setup_logging(log_file="logs/marvelous-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

