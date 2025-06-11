from loguru import logger
from marvelous.common import create_parser
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from us_accidents.config import ProjectConfig, Tags
from us_accidents.models.model import BasicModel

args = create_parser()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yaml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
model = BasicModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# Load data
model.load_data()
logger.info("Data loaded.")

# Train the model
model.train()
logger.info("Model training completed.")

# Evaluate model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)

model_improved = model.model_improved(test_set=test_set)
logger.info("Model evaluation completed, model improved: ", model_improved)

is_test = args.is_test

# when running test, always register and deploy
if is_test == 1:
    model_improved = True

if model_improved:
    # Register the model
    latest_version = model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
