"""Unit tests for DataProcessor."""

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from us_accidents.config import ProjectConfig
from us_accidents.data_processor import DataProcessor
from tests.fixtures.datapreprocessor_fixture import spark_session, config, sample_data


def test_data_ingestion(sample_data: DataFrame) -> None:
    """Test the data ingestion process by checking the shape of the sample data.

    Asserts that the sample data has at least one row and one column.

    :param sample_data: The sample data to be tested
    """
    assert sample_data.count() > 0
    assert len(sample_data.columns) > 0


def test_dataprocessor_init(
    sample_data: DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Test the initialization of DataProcessor.

    :param sample_data: Sample DataFrame for testing
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(spark=spark_session, config=config, dataframe=sample_data)
    assert isinstance(processor.dataframe, DataFrame)

    assert isinstance(processor.config, ProjectConfig)
    assert isinstance(processor.spark, SparkSession)
