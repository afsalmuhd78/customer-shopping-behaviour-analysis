import os
from itertools import chain
from dotenv import load_dotenv
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

load_dotenv()

# S3 and PostgreSQL config from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY = os.getenv("S3_KEY")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DATABASE = os.getenv("PG_DATABASE")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_TABLE = os.getenv("PG_TABLE", "customer")
PG_WRITE_MODE = os.getenv("PG_WRITE_MODE", "overwrite")

RAW_SCHEMA = StructType(
    [
        StructField("Customer ID", IntegerType(), False),
        StructField("Age", IntegerType(), True),
        StructField("Gender", StringType(), True),
        StructField("Item Purchased", StringType(), True),
        StructField("Category", StringType(), True),
        StructField("Purchase Amount (USD)", IntegerType(), True),
        StructField("Location", StringType(), True),
        StructField("Size", StringType(), True),
        StructField("Color", StringType(), True),
        StructField("Season", StringType(), True),
        StructField("Review Rating", DoubleType(), True),
        StructField("Subscription Status", StringType(), True),
        StructField("Shipping Type", StringType(), True),
        StructField("Discount Applied", StringType(), True),
        StructField("Promo Code Used", StringType(), True),
        StructField("Previous Purchases", IntegerType(), True),
        StructField("Payment Method", StringType(), True),
        StructField("Frequency of Purchases", StringType(), True),
    ]
)

FREQUENCY_DAYS_MAP = {
    "Weekly": 7,
    "Bi-Weekly": 14,
    "Fortnightly": 14,
    "Monthly": 30,
    "Quarterly": 90,
    "Every 3 Months": 90,
    "Annually": 365,
}


def get_spark_session():
    builder = (
        SparkSession.builder.appName("CustomerBehaviorETL")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,org.postgresql:postgresql:42.7.3",
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint.region", AWS_REGION)
    )

    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        builder = builder.config(
            "spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID
        ).config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
    else:
        builder = builder.config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )

    if S3_ENDPOINT:
        builder = builder.config(
            "spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT
        ).config("spark.hadoop.fs.s3a.path.style.access", "true")

    return builder.getOrCreate()


def clean_and_transform(df):
    # Standardize column names to snake_case
    for col_name in df.columns:
        new_name = col_name.strip().lower().replace(" ", "_")
        if new_name == "purchase_amount_(usd)":
            new_name = "purchase_amount"
        df = df.withColumnRenamed(col_name, new_name)

    df = df.dropDuplicates(["customer_id"])

    # Fill missing review_rating with the category median (fallback to global median)
    missing_ratings = df.filter(F.col("review_rating").isNull()).count()
    if missing_ratings > 0:
        print(f"Imputing {missing_ratings} missing review_rating values...")
        cat_window = Window.partitionBy("category")
        global_median = df.approxQuantile("review_rating", [0.5], 0.0)[0]
        cat_median = F.percentile_approx("review_rating", 0.5, 10000).over(cat_window)
        df = df.withColumn(
            "review_rating",
            F.round(
                F.coalesce(F.col("review_rating"), cat_median, F.lit(global_median)), 2
            ),
        )

    # Split age into 4 quartile groups (same idea as pandas qcut)
    q1, q2, q3 = df.approxQuantile("age", [0.25, 0.5, 0.75], 0.0)
    df = df.withColumn(
        "age_group",
        F.when(F.col("age") <= q1, "Young Adult")
        .when(F.col("age") <= q2, "Adult")
        .when(F.col("age") <= q3, "Middle-aged")
        .otherwise("Senior"),
    )

    # Map purchase frequency text to days
    mapping_expr = F.create_map(
        [F.lit(x) for x in chain(*FREQUENCY_DAYS_MAP.items())]
    )
    df = df.withColumn(
        "purchase_frequency_days",
        mapping_expr[F.col("frequency_of_purchases")].cast(IntegerType()),
    )

    # Drop promo_code_used if it matches discount_applied across all rows
    if "promo_code_used" in df.columns:
        diff_count = df.filter(
            F.col("discount_applied") != F.col("promo_code_used")
        ).count()
        if diff_count == 0:
            df = df.drop("promo_code_used")

    # Extra features for analytics
    df = (
        df.withColumn(
            "loyalty_tier",
            F.when(F.col("previous_purchases") <= 5, "New")
            .when(F.col("previous_purchases") <= 20, "Returning")
            .when(F.col("previous_purchases") <= 40, "Loyal")
            .otherwise("VIP"),
        )
        .withColumn(
            "estimated_annual_spend",
            F.round(
                F.col("purchase_amount") * (365.0 / F.col("purchase_frequency_days")), 2
            ),
        )
        .withColumn(
            "is_high_value_subscriber",
            (F.col("subscription_status") == "Yes")
            & (F.col("loyalty_tier").isin("Loyal", "VIP")),
        )
        .withColumn("ingested_at", F.current_timestamp())
    )

    return df


def check_data_quality(df):
    stats = df.agg(
        F.count("*").alias("total_rows"),
        F.sum(F.when(F.col("customer_id").isNull(), 1).otherwise(0)).alias("null_ids"),
        F.sum(F.when(F.col("review_rating").isNull(), 1).otherwise(0)).alias(
            "null_ratings"
        ),
        F.sum(F.when(F.col("purchase_amount") <= 0, 1).otherwise(0)).alias(
            "invalid_amounts"
        ),
    ).collect()[0]

    total_rows = stats["total_rows"]
    if total_rows == 0:
        raise ValueError("Data quality check failed: DataFrame is empty.")
    if stats["null_ids"] > 0:
        raise ValueError(f"Data quality check failed: {stats['null_ids']} null customer_id rows.")
    if stats["null_ratings"] > 0:
        raise ValueError(
            f"Data quality check failed: {stats['null_ratings']} null review_rating rows remain."
        )
    if stats["invalid_amounts"] > 0:
        print(f"Warning: {stats['invalid_amounts']} rows have purchase_amount <= 0.")

    print(f"Data quality checks passed ({total_rows} rows).")
    return total_rows


def load_to_postgres(df):
    jdbc_url = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

    writer = (
        df.write.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", PG_TABLE)
        .option("user", PG_USER)
        .option("password", PG_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .option("batchsize", "10000")
        .mode(PG_WRITE_MODE)
    )

    if PG_WRITE_MODE == "overwrite":
        writer = writer.option("truncate", "true")

    writer.save()


def main():
    required_vars = ["S3_BUCKET", "S3_KEY", "PG_DATABASE", "PG_USER", "PG_PASSWORD"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    spark = get_spark_session()
    try:
        s3_path = f"s3a://{S3_BUCKET}/{S3_KEY.lstrip('/')}"
        print(f"Reading data from {s3_path}...")

        raw_df = (
            spark.read.option("header", "true")
            .option("mode", "PERMISSIVE")
            .schema(RAW_SCHEMA)
            .csv(s3_path)
        )

        final_df = clean_and_transform(raw_df).cache()
        row_count = check_data_quality(final_df)

        print(f"Loading {row_count} rows into PostgreSQL table '{PG_TABLE}'...")
        load_to_postgres(final_df)
        print("Done.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()