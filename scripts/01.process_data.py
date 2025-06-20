import yaml
from loguru import logger
from marvelous.common import create_parser
from pyspark.sql import SparkSession

from us_accidents.config import ProjectConfig
from us_accidents.data_processor import DataProcessor, generate_synthetic_data, generate_test_data

args = create_parser()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yaml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
is_test = args.is_test

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the us accidents test set from the catalog
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/US_Accidents_March23.csv", header=True, inferSchema=True
).toPandas()

if is_test == 0:
    # Selects random data present in the test set
    # This is mimicking a new data arrival. In real world, this would be a new batch of data.
    # df is passed to infer schema
    new_data_pandas = generate_synthetic_data(df, num_rows=500)
    new_data = spark.createDataFrame(new_data_pandas)
    logger.info("Synthetic data generated.")
else:
    # Selects random data present in the test set
    # This is mimicking a new data arrival. This is a valid example for integration testing.
    new_data_pandas = generate_test_data(df, num_rows=500)
    new_data = spark.createDataFrame(new_data_pandas)
    logger.info("Test data generated.")

# Initialize DataProcessor
data_processor = DataProcessor(dataframe=new_data, config=config, spark=spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.printSchema())
logger.info("Test set shape: %s", X_test.printSchema())

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(train_set=X_train, test_set=X_test, write_mode="append")
