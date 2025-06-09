import pandas as pd
import numpy as np
import time

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, BooleanType, TimestampType

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

    def clean_raw_data(self) -> DataFrame:
        """Clean raw dataset, to make it ready for training and testing.

        This method performs the following steps:
        1. Selects relevant columns and filters the dataset based on the source.
        2. Cleans and transforms the data, including handling missing values and converting data types.
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
        return clean_df

    def resample_data(self, df: DataFrame, resampling_threshold: int = 15000) -> DataFrame:
        """Resample the cleaned dataset.

        Severity type 4 accident are overwhelmingly present in the dataset.
        We will resample the dataset to balance between severity 4 accidents and less severe types.
        :param dataframe: The DataFrame to be processed.
        :param resampling_threshold: Resampling threshold.
        """
        # Separate the DataFrame into two based on the "severity_4" column
        df_true = df.filter("severity_4 = true")
        df_false = df.filter("severity_4 = false")

        # Count the number of rows in each DataFrame
        count_true = df_true.count()
        count_false = df_false.count()

        # Determine the sampling fractions
        fraction_true = resampling_threshold / count_true if count_true > resampling_threshold else 1.0
        fraction_false = resampling_threshold / count_false if count_false > resampling_threshold else 1.0

        # Sample the DataFrames
        if count_true > resampling_threshold:
            df_true_sampled = df_true.sample(withReplacement=False, fraction=fraction_true, seed=42)
        else:
            df_true_sampled = df_true.sample(withReplacement=True, fraction=fraction_true, seed=42)

        if count_false > resampling_threshold:
            df_false_sampled = df_false.sample(withReplacement=False, fraction=fraction_false, seed=42)
        else:
            df_false_sampled = df_false.sample(withReplacement=True, fraction=fraction_false, seed=42)

        # Union the sampled DataFrames
        df_resampled = df_true_sampled.union(df_false_sampled)
        return df_resampled

    def preprocess(self) -> None:
        """Complete the preprocessing.

        This method performs the following steps:
        1. Creates new features based on existing data.
        2. Resamples the dataset to balance the severity of accidents.
        3. Saves the cleaned DataFrame to a Delta table in the specified catalog and schema.
        """
        clean_df = self.clean_raw_data()
        clean_df.createOrReplaceTempView("cleaned_accidents")

        # Select correct features
        featurized_df = self.spark.sql("""
            select
                case when Timezone = 'US/Eastern' then 1 else 0 end as `Timezone_US/Eastern`,
                case when Timezone = 'US/Mountain' then 1 else 0 end as `Timezone_US/Mountain`,
                case when Timezone = 'US/Pacific' then 1 else 0 end as `Timezone_US/Pacific`,
                case when Weekday = 1 then 1 else 0 end as `Weekday_1`,
                case when Weekday = 2 then 1 else 0 end as `Weekday_2`,
                case when Weekday = 3 then 1 else 0 end as `Weekday_3`,
                case when Weekday = 4 then 1 else 0 end as `Weekday_4`,
                case when Weekday = 5 then 1 else 0 end as `Weekday_5`,
                case when Weekday = 6 then 1 else 0 end as `Weekday_6`,
                cast(Station as int) as Station,
                cast(Stop as int) as Stop,
                cast(Traffic_Signal as int) as Traffic_Signal,
                case when Severity = 4 then 1 else 0 end as `Severity_4`,
                case when contains(Street, 'Rd ') then 1 else 0 end as `Rd`,
                case when contains(Street, 'St ') then 1 else 0 end as `St`,
                case when contains(Street, 'Dr ') then 1 else 0 end as `Dr`,
                case when contains(Street, 'Ave ') then 1 else 0 end as `Ave`,
                case when contains(Street, 'Blvd ') then 1 else 0 end as `Blvd`,
                case when contains(Street, 'I- ') then 1 else 0 end as `I-`,
                case when Astronomical_Twilight = 'Night' then 1 else 0 end as Astronomical_Twilight_Night,
                Start_Lat,
                Start_Lng,
                Pressure_in as Pressure_bc
            from cleaned_accidents
        """)

        # Resample the dataset: over-presence of severity 4 accidents
        resampled_df = self.resample_data(featurized_df)

        # Save this dataframe to a UC table
        resampled_df_with_timestamp = resampled_df.withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        resampled_df_with_timestamp.write.mode("overwrite").saveAsTable(self.clean_df_address)

    def split_data(self, test_size: float = 0.3, seed: int = 42) -> tuple[DataFrame, DataFrame]:
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

        train_set_with_timestamp.write.mode("append").saveAsTable(self.training_set_address)
        test_set_with_timestamp.write.mode("append").saveAsTable(self.test_set_address)

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

def generate_synthetic_data(df: DataFrame, num_rows: int = 500) -> DataFrame:
    """Generate synthetic data matching input DataFrame distributions.

    Creates artificial dataset replicating statistical patterns from source columns including numeric,
    categorical, and datetime types.

    :param df: Source DataFrame containing original data distributions
    :param num_rows: Number of synthetic records to generate
    :return: DataFrame containing generated synthetic data
    """
    spark = df.sparkSession
    synthetic_data = pd.DataFrame()
    df = df.toPandas()  # Convert Spark DataFrame to Pandas DataFrame for processing

    # Convert relevant numeric columns to float64
    # This is necessary for compatibility with PySpark DataFrame creation
    int_columns = {
        "Start_Lat",
        "Start_Lng",
        "Pressure_bc"
    }
    for col in int_columns.intersection(df.columns):
        synthetic_data[col] = synthetic_data[col].astype(np.float64)

    for column in df.columns:
        if column == "Id":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    timestamp_base = int(time.time() * 1000)
    synthetic_data["Id"] = [str(timestamp_base + i) for i in range(num_rows)]

    synthetic_data_spark = spark.createDataFrame(synthetic_data)
    return synthetic_data_spark