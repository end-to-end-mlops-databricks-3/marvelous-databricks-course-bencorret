from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, current_timestamp, lit, lower, to_utc_timestamp, when

from us_accidents.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig, dataframe: DataFrame) -> None:
        """Initialize the DataProcessor with a Spark session, configuration, and DataFrame.
        :param spark: Spark session to be used for DataFrame operations.
        :param config: Configuration object containing catalog and schema information.
        :param dataframe: The DataFrame to be processed.
        """
        self.dataframe = dataframe
        self.config = config
        self.spark = spark
        self.clean_df_address = f"{self.config.catalog_name}.{self.config.schema_name}.{self.config.cleaned_table_name}"
        self.training_set_address = f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        self.test_set_address = f"{self.config.catalog_name}.{self.config.schema_name}.test_set"

    def preprocess(self) -> None:
        """Clean raw dataset, to make it ready for training and testing.
        This method performs the following steps:
        1. Selects relevant columns and filters the dataset based on the source.
        2. Cleans and transforms the data, including handling missing values and converting data types.
        3. Creates new features based on existing data.
        4. Saves the cleaned DataFrame to a Delta table in the specified catalog and schema.
        """
        self.dataframe.createOrReplaceTempView("raw_accidents")

        # Features 'ID' doesn't provide any useful information about accidents themselves.
        # 'TMC', 'Distance(mi)', 'End_Time' (we have start time), 'Duration', 'End_Lat', and 'End_Lng'(we have start location)
        # can be collected only after the accident has already happened and hence cannot be predictors
        # for serious accident prediction.
        # For 'Description', the POI features have already been extracted from it by dataset creators.
        # 'Country' and 'Turning_Loop' dropped too, for they have only one class.
        # More than 60% percent of 'Number', 'Wind_Chill(F)' is missing; we drop these columns too.

        # 1 - We arbitrarily choose one source only. Different websites / databases use different collection and classification methodologies.
        # We will focus on the source offering the most data: Source1
        clean_df = self.spark.sql("""
            WITH median_precipitation AS (
            SELECT median(CAST(`Precipitation(in)` as decimal(12,4))) AS median_precip
            FROM raw_accidents
            WHERE `Precipitation(in)` IS NOT NULL
            )
            SELECT
                CAST(`Severity` as string) as `Severity`,
                CAST(`Start_Time` as timestamp) as `Start_Time`,
                CAST(extract(YEAR FROM CAST(`Start_Time` as timestamp)) as int) as `Year`,
                CAST(extract(MONTH FROM CAST(`Start_Time` as timestamp)) as int) as `Month`,
                CAST(WEEKDAY(CAST(`Start_Time` as timestamp)) as int) as `Weekday`,
                CAST(extract(DAY FROM CAST(`Start_Time` as timestamp)) as int) as `Day`,
                CAST(extract(HOUR FROM CAST(`Start_Time` as timestamp)) as int) as `Hour`,
                CAST(extract(MINUTE FROM CAST(`Start_Time` as timestamp)) as int) as `Minute`,
                CAST(`Start_Lat` as decimal(38, 10)) as `Start_Lat`,
                CAST(`Start_Lng` as decimal(38, 10)) as `Start_Lng`,
                CAST(`Street` as string) as `Street`,
                CAST(`City` as string) as `City`,
                CAST(`County` as string) as `County`,
                CAST(`State` as string) as `State`,
                CAST(`Zipcode` as string) as `Zipcode`,
                CAST(`Timezone` as string) as `Timezone`,
                CAST(`Airport_Code` as string) as `Airport_Code`,
                CAST(`Weather_Timestamp` as timestamp) as `Weather_Timestamp`,
                CAST(`Temperature(F)` as decimal(38, 10)) as `Temperature_F`,
                CAST(`Humidity(%)` as decimal(38, 10)) as `Humidity_%`,
                CAST(`Pressure(in)` as decimal(38, 10)) as `Pressure_in`,
                CAST(`Visibility(mi)` as decimal(38, 10)) as `Visibility_mi`,
                CASE
                    WHEN Wind_Direction = 'Calm' THEN 'CALM'
                    WHEN Wind_Direction = 'West' THEN 'W'
                    WHEN Wind_Direction = 'WSW' THEN 'W'
                    WHEN Wind_Direction = 'South' THEN 'S'
                    WHEN Wind_Direction = 'SSW' THEN 'S'
                    WHEN Wind_Direction = 'North' THEN 'N'
                    WHEN Wind_Direction = 'NNW' THEN 'N'
                    WHEN Wind_Direction = 'East' THEN 'E'
                    WHEN Wind_Direction = 'ESE' THEN 'E'
                    WHEN Wind_Direction = 'Variable' THEN 'VAR'
                    ELSE Wind_Direction
                END as `Wind_Direction`,
                CAST(`Wind_Speed(mph)` as decimal(38, 10)) as `Wind_Speed_mph`,
                CASE
                    WHEN `Precipitation(in)` IS NULL THEN (SELECT median_precip FROM median_precipitation)
                    ELSE CAST(`Precipitation(in)` as decimal(38, 10))
                END as `Precipitation_in`,
                CAST(`Weather_Condition` as string) as `Weather_Condition`,
                CAST(`Amenity` as boolean) as `Amenity`,
                CAST(`Bump` as boolean) as `Bump`,
                CAST(`Crossing` as boolean) as `Crossing`,
                CAST(`Give_Way` as boolean) as `Give_Way`,
                CAST(`Junction` as boolean) as `Junction`,
                CAST(`No_Exit` as boolean) as `No_Exit`,
                CAST(`Railway` as boolean) as `Railway`,
                CAST(`Roundabout` as boolean) as `Roundabout`,
                CAST(`Station` as boolean) as `Station`,
                CAST(`Stop` as boolean) as `Stop`,
                CAST(`Traffic_Calming` as boolean) as `Traffic_Calming`,
                CAST(`Traffic_Signal` as boolean) as `Traffic_Signal`,
                CAST(`Sunrise_Sunset` as string) as `Sunrise_Sunset`,
                CAST(`Civil_Twilight` as string) as `Civil_Twilight`,
                CAST(`Nautical_Twilight` as string) as `Nautical_Twilight`,
                CAST(`Astronomical_Twilight` as string) as `Astronomical_Twilight`
            FROM raw_accidents
            WHERE Source = 'Source1'
        """)

        # 2 - Special treatment of weather conditions
        # We first simplify the weather conditions to a few categories
        clean_df = clean_df.withColumn(
            "Clear", when(lower(col("Weather_Condition")).contains("clear"), True).otherwise(False)
        )
        clean_df = clean_df.withColumn(
            "Cloud",
            when(
                lower(col("Weather_Condition")).contains("cloud")
                | lower(col("Weather_Condition")).contains("overcast"),
                True,
            ).otherwise(False),
        )
        clean_df = clean_df.withColumn(
            "Rain",
            when(
                lower(col("Weather_Condition")).contains("rain") | lower(col("Weather_Condition")).contains("storm"),
                True,
            ).otherwise(False),
        )
        clean_df = clean_df.withColumn(
            "Heavy_Rain",
            when(
                lower(col("Weather_Condition")).contains("heavy rain")
                | lower(col("Weather_Condition")).contains("rain shower")
                | lower(col("Weather_Condition")).contains("heavy t-storm")
                | lower(col("Weather_Condition")).contains("heavy thunderstorms"),
                True,
            ).otherwise(False),
        )
        clean_df = clean_df.withColumn(
            "Snow",
            when(
                lower(col("Weather_Condition")).contains("snow")
                | lower(col("Weather_Condition")).contains("sleet")
                | lower(col("Weather_Condition")).contains("ice"),
                True,
            ).otherwise(False),
        )
        clean_df = clean_df.withColumn(
            "Heavy_Snow",
            when(
                lower(col("Weather_Condition")).contains("heavy snow")
                | lower(col("Weather_Condition")).contains("heavy sleet")
                | lower(col("Weather_Condition")).contains("heavy ice pellets")
                | lower(col("Weather_Condition")).contains("snow showers")
                | lower(col("Weather_Condition")).contains("squalls"),
                True,
            ).otherwise(False),
        )
        clean_df = clean_df.withColumn(
            "Fog", when(lower(col("Weather_Condition")).contains("fog"), True).otherwise(False)
        )

        # And show weather condition fields as NULLs if the Weather_Condition field is NULL
        weather = ["Clear", "Cloud", "Rain", "Heavy_Rain", "Snow", "Heavy_Snow", "Fog"]
        for condition in weather:
            clean_df = clean_df.withColumn(
                condition, when(col("Weather_Condition").isNull(), lit(None)).otherwise(col(condition))
            )

        # 3 - Drop rows with null values in the specified columns
        clean_df = clean_df.dropna(
            subset=[
                "City",
                "Zipcode",
                "Airport_Code",
                "Sunrise_Sunset",
                "Civil_Twilight",
                "Nautical_Twilight",
                "Astronomical_Twilight",
            ]
        )

        # 4 - Create instance variable with cleaned DataFrame
        clean_df_with_timestamp = clean_df.withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        clean_df_with_timestamp.write.mode("overwrite").saveAsTable(self.clean_df_address)

    def split_data(self, test_size: float = 0.2, seed: int = 42) -> tuple[DataFrame, DataFrame]:
        """Split the DataFrame (self.clean_df) into training and test sets using PySpark's randomSplit.

        :param test_size: The proportion of the dataset to include in the test split.
        :param seed: Seed for random number generation to ensure reproducibility.
        :return: A tuple containing the training and test DataFrames.
        """
        # Calculate the weights for train and test splits
        train_size = 1.0 - test_size
        cleaned_table = self.spark.read.table(self.clean_df_address)
        train_df, test_df = cleaned_table.randomSplit([train_size, test_size], seed=seed)
        return train_df, test_df

    def save_to_catalog(self, train_set: DataFrame, test_set: DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = train_set.withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_set_with_timestamp = test_set.withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(self.training_set_address)
        test_set_with_timestamp.write.mode("overwrite").saveAsTable(self.test_set_address)

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
