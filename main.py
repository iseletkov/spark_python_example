from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# https://bigdataschool.ru/blog/linear-regression-in-pyspark.html
# https://ruslanmv.com/blog/Linear-Regression-with-Python-and-Spark

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

    # Загрузка исходных данных из СУБД
    courses = spark.read \
        .format("jdbc") \
        .option("url", f"jdbc:postgresql://{host_db}:{port_db}/{database}") \
        .option("dbtable", schema + "." + table_courses) \
        .option("user", user) \
        .option("password", password) \
        .option("driver", "org.postgresql.Driver") \
        .load() \
        .filter(col("num_code").isin([36, 156, 356, 840, 944, 978])) \
        .withColumn("value", col("value").cast("double")) \
        .groupBy("dttm") \
        .pivot("num_code") \
        .sum("value") \
        .orderBy("dttm")

    features = ["36", "156", "356", "944", "978"]
    target = "840"

    # Для обучения модели все исходные колонки собираются в одну,
    # содержащую вектор из значений всех остальных колонок.
    assembler = VectorAssembler(inputCols=features,
                                outputCol='features')
    courses_vectorized = assembler.transform(courses)

    print("courses_vectorized:")
    courses_vectorized.show()

    # Деление на тестовую и обучающую выборки.
    train, test = courses_vectorized.randomSplit([0.7, 0.3])

    # Построение линейной регрессии
    lr = LinearRegression(
        featuresCol='features',
        labelCol=target)

    model = lr.fit(train, )

    # Тестирование, расчёт метрик.
    predicted = model.evaluate(test)
    predicted.predictions.show()

    # Вывод результата.
    print(f"RMSE: {predicted.rootMeanSquaredError}")
    print(f"MAE: {predicted.meanAbsoluteError}")

    # model.write().overwrite().save("model")


main()
