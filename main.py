from pyspark.sql import SparkSession

host_spark = "localhost"
host_db = "192.168.1.102"
port_spark = 8081
port_db = 50000
database = "test"
user = "test"
password = "test"
schema = "public"
table_courses = "courses"
table_currencies = "currencies"


def main():
    spark = SparkSession \
        .builder \
        .master(f"spark://{host_spark}:{port_spark}") \
        .config("spark.jars", "postgresql-42.7.1.jar") \
        .appName("Test_db") \
        .getOrCreate()

    courses = spark.read \
        .format("jdbc") \
        .option("url", f"jdbc:postgresql://{host_db}:{port_db}/{database}") \
        .option("dbtable", schema + "." + table_courses) \
        .option("user", user) \
        .option("password", password) \
        .option("driver", "org.postgresql.Driver") \
        .load()

    df = courses.toPandas()
    df.to_excel("report.xlsx")


main()

