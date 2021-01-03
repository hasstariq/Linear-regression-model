
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext
from numpy import array
from pyspark.mllib.linalg import Vectors

sc = SparkContext()

data = MLUtils.loadLibSVMFile(sc, "C:\Users\hassan\Desktop\422a3\train_scaled.txt")


mod = LinearRegressionWithSGD.train(data, iterations = 100, intercept = True)

w = "Weights: " + str(mod.weights)
i = "Intercept: " + str(mod.intercept)
print(w)

print(i)