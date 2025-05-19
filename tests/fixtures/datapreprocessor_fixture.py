"""Dataloader fixture."""

import pytest
from loguru import logger
from pyspark.sql import DataFrame, SparkSession

from tests.unit_tests.spark_config import spark_config
from us_accidents import PROJECT_DIR
from us_accidents.config import ProjectConfig, Tags


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Create and return a SparkSession for testing.

    This fixture creates a SparkSession with the specified configuration and returns it for use in tests.
    """
    # One way
    # spark = SparkSession.builder.getOrCreate()  # noqa
    # Alternative way - better
    spark = (
        SparkSession.builder.master(spark_config.master)
        .appName(spark_config.app_name)
        .config("spark.executor.cores", spark_config.spark_executor_cores)
        .config("spark.executor.instances", spark_config.spark_executor_instances)
        .config("spark.sql.shuffle.partitions", spark_config.spark_sql_shuffle_partitions)
        .config("spark.driver.bindAddress", spark_config.spark_driver_bindAddress)
        .getOrCreate()
    )

    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load and return the project configuration.

    This fixture reads the project configuration from a YAML file and returns a ProjectConfig object.

    :return: The loaded project configuration
    """
    config_file_path = (PROJECT_DIR / "project_config.yaml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")
    config = ProjectConfig.from_yaml(config_file_path.as_posix())
    return config


@pytest.fixture(scope="function")
def sample_data(config: ProjectConfig, spark_session: SparkSession) -> DataFrame:
    """Create a sample DataFrame from a CSV file.

    This fixture reads a CSV file using either Spark or pandas, then converts it to a Spark DataFrame,

    :return: A sampled Spark DataFrame containing some sample of the original data.
    """
    file_path = PROJECT_DIR / "tests" / "test_data" / "sample.csv"
    sample = (
        spark_session.read.format("csv").option("header", "true").option("separator", ",").load(file_path.as_posix())
    )
    return sample


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Create and return a Tags instance for the test session.

    This fixture provides a Tags object with predefined values for git_sha, branch, and job_run_id.
    """
    return Tags(git_sha="wxyz", branch="test", job_run_id="9")
