<html>
<h2 align="center" > Final Project Document</h2>

<div  align="center" >

![ScreenShot](https://github.com/JesuaMG/git_flow/blob/Features/portada.png)

</div>

<h3align="center" >Instituto Nacional de México</h3>
<h3align="center" >Instituto Tecnológico de Tijuana</h3>
<h3align="center" >Departamento de sistemas y computación</h3>
<br>
<h4 align="center" >Ingeniería en Sistemas Computacionales</h4>
<h4 align="center" >Semester of Sep 2020 - Ene 2021</h4>
<br>
<h3 align="center" >Datos Masivos </h3>
<br>
<h4 align="center" >Unit 4</h4>
<h4 align="center" >Final Project</h4>
<br>
<h4 align="center" >Made by : </h4>
<h4 align="center" >Hernández Negrete Juan Carlos - 16212021</h4>
<h4 align="center" >Manzano Guzmán Jesua - 16212033</h4>
<br>
<h4 align="center" >Professor : </h4>
<h4 align="center" >Dr. Jose Christian Romero Hernandez</h4>
<br>
<h4 align="center" >Delivery date: 2021 / January / 15 </h4>
</html>

## Index
- [Index](#index)
- [Introduction](#introduction)
- [Theoretical framework of algorithms](#theoretical-framework-of-algorithms)
  - [SVM](#svm)
  - [Decision Tree](#decision-tree)
- [Implementation](#implementation)
  - [SVM (Support Vector Machine)](#svm-support-vector-machine)
  - [Decision Tree](#decision-tree-1)
- [Results](#results)
- [Conclusions](#conclusions)
- [References](#references)

## Introduction

<p align = "justify">
The purpose of this document is to analyze, compare and verify the efficiency of 4 classification algorithms, which are SVM (Support Vector Machine), Decision Tree, Logistic Regresion and Multilayer Perceptron, working with the bank-full.csv dataset (https: //archive.ics.uci.edu/ml/datasets/Bank+Marketing).
<br>
To carry out the development of this project, it will be necessary to implement the knowledge acquired in the course of Big Data.


## Theoretical framework of algorithms

### SVM
<p align = "justify">
A support vector machine (SVM) is a supervised learning algorithm that can be used for binary classification or regression. Support vector machines are very popular in applications such as natural language processing, speech, image recognition, and computer vision.<br>
A support vector machine constructs an optimal hyperplane in the form of a decision surface, so that the margin of separation between the two classes in the data is maximally widened. Support vectors refer to a small subset of the training observations that are used as support for the optimal location of the decision surface. <br>
Support vector machines belong to a class of Machine Learning algorithms called kernel methods and are also known as kernel machines. 

![ScreenShot](https://github.com/JesuaMG/git_flow/blob/Features/1607682266241.jpg)

The training of a support vector machine consists of two phases:

- Transform the predictors (input data) into a highly dimensional feature space. In this phase it is enough to specify the kernel; data is never explicitly transformed into feature space. This process is commonly known as the kernel hack. 
- Solve a quadratic optimization problem that fits an optimal hyperplane to classify the transformed features into two classes. The number of transformed features is determined by the number of support vectors. 

### Decision Tree
<p align ="justify"> 
Decision tree is the most powerful and popular tool for classification and prediction. A Decision tree is a flowchart like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label. <br><br>
Decision trees classify instances by sorting them down the tree from the root to some leaf node, which provides the classification of the instance. An instance is classified by starting at the root node of the tree,testing the attribute specified by this node,then moving down the tree branch corresponding to the value of the attribute as shown in the above figure.This process is then repeated for the subtree rooted at the new node. 
<br><br>

![ScreenShot](https://github.com/JesuaMG/git_flow/blob/Features/DT.png)

The strengths of decision tree methods are: 

- Decision trees are able to generate understandable rules.
- Decision trees perform classification without requiring much computation.
- Decision trees are able to handle both continuous and categorical variables.
- Decision trees provide a clear indication of which fields are most important for prediction or classification.

## Implementation

<p align="justify"> To carry out this project Apache Spark was used with the Scala programming language, it was decided to use this FrameWork due to its efficiency for Big Data, in addition to the fact that it can be used with 3 different programming languages (Python, Java and Scala), it was used Scala due to the many advantages it offers such as the easy scalability of the code, it is a language based on the Object Oriented paradigms and the Functional paradigm, it can be run in Java Virtual Machine, it is also faster than Python and Java. Another advantage that we could find in Apache Spark is the documentation of this platform which allowed us to resolve doubts throughout the development of this project.


### SVM (Support Vector Machine)

Import libs
```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
``` 
Minimize errors and  start a simple Spark session.
```scala
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder.appName("svm").getOrCreate()
```
Load the bank-full.csv dataset and check the data
```scala
val df  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

df.head()
df.describe()
```
Index column "y", Create a vector with the columns with numerical data and name it as features, Use the assembler object to transform features
```scala
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedY").fit(df)
val indexed = labelIndexer.transform(df).drop("y").withColumnRenamed("indexedY", "label")

val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))

val featurestrans = vectorFeatures.transform(indexed)
```
Rename column "y" as label, Union of label and features as dataIndexed, Creation of labelIndexer and featureIndexer for the pipeline
```scala
val featureslabel = featurestrans.withColumnRenamed("y", "label")

val dataindexed = featureslabel.select("label","features")
dataindexed.show()

val labelindexer = new StringIndexer().setInputCol("label").setOutputCol("indexedlabel").fit(dataindexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedfeatures").setMaxCategories(4).fit(dataindexed)
```
Training data as 70% and test data as 30%.
```scala
val Array(training, test) = dataindexed.randomSplit(Array(0.7, 0.3), seed = 1234L)
```
Linear Support Vector Machine object, Fitting the trainingData into the model, Transforming testData for the predictions.
```scala
val supportVM = new LinearSVC().setMaxIter(10).setRegParam(0.1)
    
val modelSVM = supportVM.fit(training)

val predictions = modelSVM.transform(test)
```
Obtaining the metrics, Confusion matrix, Accuracy and Test Error.
```scala
val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

println("Confusion matrix:")
println(metrics.confusionMatrix)

println("Accuracy: " + metrics.accuracy) 
println(s"Test Error = ${(1.0 - metrics.accuracy)}")
```

### Decision Tree

Import libs
```scala
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.StringIndexer 
import org.apache.spark.ml.Pipeline
```
Minimize errors and  start a simple Spark session.
```scala
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder.appName("svm").getOrCreate()
```
Load the bank-full.csv dataset and check the data
```scala
val df  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

df.head()
df.describe()
```
Transform categorical data to numeric, Create a vector with the columns with numerical data and name it as features, Use the assembler object to transform features
```scala
val stringindexer = new StringIndexer().setInputCol("y").setOutputCol("label")
val df = stringindexer.fit(dataframe).transform(dataframe)

val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","campaign","pdays","previous")).setOutputCol("features")
val output = assembler.transform(df)

val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(output)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(output)
```
Training data as 70% and test data as 30%
```scala
val Array(trainingData, testData) = output.randomSplit(Array(0.7, 0.3), seed = 1234L)
```
Train a model, Convert indexed labels back to original labels, Chain indexers and tree in a Pipeline
```scala
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
```
Train model, Make predictions
```scala
val model = pipeline.fit(trainingData)

val predictions = model.transform(testData)

predictions.select("predictedLabel", "label", "features").show(5)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")
```


## Results

## Conclusions

## References
> Marks, R. J., Moulin, L. S., da Silva, A. A., & El-Sharkawi, M. A. (2001). Neural networks and support vector machines applied to power systems transient stability analysis. International journal of engineering intelligent systems for electrical engineering and communications, 9(4), 205-212.

> Viera, Á. F. G. (2017). Técnicas de aprendizaje de máquina utilizadas para la minería de texto. Investigación bibliotecológica, 31(71), 103-126.

> Sana, B., Siddiqui, I. F., & Arain, Q. A. (2019). Analyzing students’ academic performance through educational data mining.