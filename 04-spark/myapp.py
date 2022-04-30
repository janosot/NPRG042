from __future__ import print_function

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("/opt/data/seznam.csv", header=False)
df = df.toDF('first_name', 'last_name', 'id', 'zip')

df.createOrReplaceTempView("persons")

sqlDF = spark.sql("SELECT r.region, SUM(r.count) as count FROM (SELECT COUNT(*) as count, LEFT (zip, 1) AS region FROM persons GROUP BY region, first_name, last_name) AS r WHERE r.count > 1 GROUP BY region ORDER BY region ASC")
with open("/mnt/1/output.csv", "w") as f:
    for row in sqlDF.rdd.collect():
        s = str(row["region"]) + "," + str(row["count"]) + "\n"
        f.write(s)
