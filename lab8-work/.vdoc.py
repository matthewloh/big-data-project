# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import split, count, when, isnan, col, regexp_replace
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, FloatType
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore') 
from pyspark.sql import SparkSession 
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, FloatType 
from pyspark.sql.functions import split, count, when, isnan, col, regexp_replace 
from pyspark.ml.regression import LinearRegression 
from pyspark.ml.feature import OneHotEncoder, StringIndexer 
from pyspark.ml.linalg import Vectors 
from pyspark.ml.feature import VectorAssembler
import setuptools.dist

spark = SparkSession.builder.appName('First Session').getOrCreate() 
print('Spark Version: {}'.format(spark.version))
#
#
#
#
# Defining a Schema
schema = StructType([StructField('mpg', FloatType(), nullable=True),
                     StructField('cylinders', IntegerType(), nullable=True),
                     StructField('displacement', FloatType(), nullable=True),
                     StructField('horsepower', StringType(), nullable=True),
                     StructField('weight', IntegerType(), nullable=True),
                     StructField('acceleration', FloatType(), nullable=True),
                     StructField('model year', IntegerType(), nullable=True),
                     StructField('origin', IntegerType(), nullable=True),
                     StructField('car name', StringType(), nullable=True)])

# Data is from UCI Machine Learning Repository
# See the detail in https://archive.ics.uci.edu/ml/datasets/auto+mpg
file_path = 'auto-mpg.csv'
df = spark.read.csv(file_path,
                    header=True,
                    inferSchema=True,
                    nanValue='?')
df.show(5)
```
#
# Check Missing Values
def check_missing(dataframe):
    return dataframe.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in dataframe.columns]).show()


check_missing(df)
#
#
#
# Handling Missing Values
df = df.na.drop()

# convert horsepower from string to int
df = df.withColumn("horsepower", df["horsepower"].cast(IntegerType()))
df.show(5)
#
#
#
# Check column names
df.columns
# Display data with pandas format
df.toPandas().head()
#
#
#
#
# Check the schema
df.printSchema()
#
#
#
# Renaming Columns
df = df.withColumnRenamed('model year', 'model_year')
df = df.withColumnRenamed('car name', 'car_name')
df.show(3)
#
#
#
# Get infos from first 4 rows
for car in df.head(4):
    print(car, '\n')
#
#
#
# statistical summary of dataframe
df.describe().show()
#
#
#
# describe with specific variables
df.describe(['mpg', 'horsepower']).show()
#
#
#
# describe with numerical columns
def get_num_cols(dataframe):
    num_cols = [col for col in dataframe.columns if dataframe.select(
        col).dtypes[0][1] in ['double', 'int']]
    return num_cols


num_cols = get_num_cols(df)
df.describe(num_cols).show()
#
#
#
#
# Lets get the cars with mpg more than 23
df.filter(df['mpg'] > 23).show(5)
```
#
# Multiple Conditions
df.filter((df['horsepower'] > 80) &
          (df['weight'] > 2000)).select('car_name').show(5)
#
#
#
# Get the cars with 'volkswagen' in their names, and sort them by model year and horsepower
df.filter(df['car_name'].contains('volkswagen')).orderBy(
    ['model_year', 'horsepower'], ascending=[False, False]).show(5)
```
#
df.filter(df['car_name'].like('%volkswagen%')).show(3)
```
#
# Get the cars with 'toyota' in their names
df.filter("car_name like '%toyota%'").show(5)
```
#
df.filter('mpg > 22').show(5)
```
#
# Multiple Conditions
df.filter('mpg > 22 and acceleration < 15').show(5)
```
#
df.filter('horsepower == 88 and weight between 2600 and 3000').select(
    ['horsepower', 'weight', 'car_name']).show()
```
#
# Brands
df.createOrReplaceTempView('auto_mpg')
df = df.withColumn('brand', split(
    df['car_name'], ' ').getItem(0)).drop('car_name')
```
#
# Replacing Misspelled Brands
auto_misspelled = {'chevroelt': 'chevrolet',
                   'chevy': 'chevrolet',
                   'vokswagen': 'volkswagen',
                   'vw': 'volkswagen',
                   'hi': 'harvester',
                   'maxda': 'mazda',
                   'toyouta': 'toyota',
                   'mercedes-benz': 'mercedes'}
for key in auto_misspelled.keys():
    df = df.withColumn('brand', regexp_replace(
        'brand', key, auto_misspelled[key]))
df.show(5)
#
#
#
# Avg Acceleration by car brands
df.groupBy('brand').agg({'acceleration': 'mean'}).show(5)
# Max MPG by car brands
df.groupBy('brand').agg({'mpg': 'max'}).show(5)
#
#
#
#
# Check brand frequencies first
df.groupby('brand').count().orderBy('count', ascending=False).show(5)
```
#
def one_hot_encoder(dataframe, col):

    # converting categorical values into category indices
    indexed = StringIndexer().setInputCol(col).setOutputCol(
        col + '_cat').fit(dataframe).transform(dataframe)
    ohe = OneHotEncoder().setInputCol(col + '_cat').setOutputCol(col +
                        '_OneHotEncoded').fit(indexed).transform(indexed)

    ohe = ohe.drop(*[col, col + '_cat'])
    return ohe

df = one_hot_encoder(df, col='brand')
df.show(5)
```
#
# Vector Assembler
def vector_assembler(dataframe, indep_cols):

    assembler = VectorAssembler(inputCols=indep_cols,
                                outputCol='features')
    output = assembler.transform(dataframe).drop(*indep_cols)

    return output


df = vector_assembler(df, indep_cols=df.drop('mpg').columns)
df.show(5) 
#
#
#
#
train_data, test_data = df.randomSplit([0.8, 0.2])
print('Train Shape: ({}, {})'.format(
    train_data.count(), len(train_data.columns)))
print('Test Shape: ({}, {})'.format(test_data.count(), len(test_data.columns)))
#
#
#
lr = LinearRegression(labelCol='mpg',
                      featuresCol='features',
                      regParam=0.3)  # avoid overfitting
lr = lr.fit(train_data)
```
#
def evaluate_reg_model(model, test_data):

    print(model.__class__.__name__.center(70, '-'))
    model_results = model.evaluate(test_data)
    print('R2: {}'.format(model_results.r2))
    print('MSE: {}'.format(model_results.meanSquaredError))
    print('RMSE: {}'.format(model_results.rootMeanSquaredError))
    print('MAE: {}'.format(model_results.meanAbsoluteError))
    print(70*'-')


evaluate_reg_model(lr, test_data)
#
#
#
# End Session
spark.stop()
#
#
#
