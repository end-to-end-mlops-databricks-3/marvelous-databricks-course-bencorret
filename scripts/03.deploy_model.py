from databricks.sdk import WorkspaceClient
from loguru import logger
from marvelous.common import create_parser
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from us_accidents.config import ProjectConfig
from us_accidents.serving.model_serving import ModelServing

args = create_parser()

root_path = args.root_path
is_test = args.is_test
config_path = f"{root_path}/files/project_config.yml"

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = config.model_names["model1"]
endpoint_name = f"{config.endpoint_names['endpoint1']}-{args.env}"

# Initialize Model Serving
logger.info("Initializing ModelServing class ...")
model_serving = ModelServing(model_name=f"{catalog_name}.{schema_name}.{model_name}", endpoint_name=endpoint_name)

# Deploy the model serving endpoint
logger.info("Deploy the model serving endpoint ...")
model_serving.deploy_or_update_serving_endpoint()
logger.info("Completed the deployment of the model serving endpoint.")

# Delete endpoint if test
if is_test == 1:
    workspace = WorkspaceClient()
    workspace.serving_endpoints.delete(name=endpoint_name)
    logger.info("Deleting serving endpoint.")
