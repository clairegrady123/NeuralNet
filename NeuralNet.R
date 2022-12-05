#2a
#set working directory
#setwd("/Users/clairegrady/Desktop/UNE/Statistical Learning/Assignment 5")
#load Toy data set
Toy = read.csv("Toy.csv", header = T)
#view data set
View(Toy)
#install neauralnet package
install.packages("neuralnet")
#load neuralnet library
library(neuralnet)
#check for NAs
which(is.na(Toy))
#attach predictor names
attach(Toy)
#view summary of the Toy data set
summary(Toy)
#load dplyr library
library(dplyr)
#checking for normality
apply(Toy,2,shapiro.test)
#load caTools library
library(caTools)

#set seed for reproducibility
set.seed(430)
#randomly split the data into two parts, 75% for training and 25% for testing
sample = sample.split(Toy$X, SplitRatio = 0.75) 
#assign the 75% for training data
trainingToy = subset(Toy, sample == TRUE) 
#assign the remaining 25% for test data
testToy = subset(Toy, sample == FALSE)
#check the number of rows to make sure training data has split correctly
nrow(trainingToy)
#check the number of rows to make sure testing data has split correctly
nrow(testToy)

#create a min-max normalize function
normalize = function(x) {
  return((x - min(x)) / (max(x) - min(x)))} 
#apply the normalize function to all columns in the trainingToy data set 
trainToyNorm = as.data.frame(lapply(trainingToy, normalize))
#apply the normalize function to all columns in the testToy data set
testToyNorm = as.data.frame(lapply(testToy, normalize))
#view summary of trainingToy data set
summary(trainingToy)
#view summary of testToy data set
summary(testToy)
#view summary of trainingToyNorm data set
summary(trainToyNorm)
#view summary of testToyNorm data set
summary(testToyNorm)

#set seed for reproducibility
set.seed(430)
#train neuralnet using backprop algorithm and logistic activation function
nnBackLog = neuralnet(Y ~., trainToyNorm, algorithm = "backprop", hidden = 1, linear.output = FALSE, act.fct = "logistic", learningrate = 0.0001, stepmax = 1e7)
#view neuralnet using backprop algorithm and logistic activation function
nnBackLog
#use neural net to predict with test data
nnBackLogPredict = neuralnet::compute(nnBackLog, testToyNorm)
#compute the correlation coefficient
nnBackLogCor = cor(nnBackLogPredict$net.result, testToyNorm$Y)
#view the correlation coefficient
nnBackLogCor
#train neuralnet using backprop algorithm and tanh activation function
nnBackTanh = neuralnet(Y ~., trainToyNorm, algorithm = "backprop", hidden = 1, linear.output = FALSE, act.fct = "tanh", learningrate = 0.0001, stepmax = 1e7)
#view neuralnet using backprop algorithm and logistic activation function
nnBackTanh
#use neural net to predict with test data
nnBackTanhPredict = neuralnet::compute(nnBackTanh, testToyNorm)
#compute the correlation coefficient
nnBackTanhCor = cor(nnBackTanhPredict$net.result, testToyNorm$Y)
#view the correlation coefficient
nnBackTanhCor
#train neuralnet using backprop algorithm and default(sigmoid) activation function
nnBackSig = neuralnet(Y ~., trainToyNorm, algorithm = "backprop", hidden = 1, learningrate = 0.0001, linear.output = TRUE, stepmax = 1e7)
#view neuralnet using backprop algorithm and default(sigmoid) activation function
nnBackSig
#use neural net to predict with test data
nnBackSigPredict = neuralnet::compute(nnBackSig, testToyNorm)
#compute the correlation coefficient
nnBackSigCor = cor(nnBackSigPredict$net.result, testToyNorm$Y)
#view the correlation coefficient
nnBackSigCor

#set seed for reproducibility
set.seed(430)
#train neuralnet using rprop+ algorithm and logistic activation function
nnRPLog = neuralnet(Y ~., trainToyNorm, algorithm = "rprop+", hidden = 1, linear.output = FALSE, act.fct = "logistic", stepmax = 1e7)
#view neuralnet using rprop+ algorithm and logistic activation function
nnRPLog
#use neural net to predict with test data
nnRPLogPredict = neuralnet::compute(nnRPLog, testToyNorm)
#compute the correlation coefficient
nnRPLogCor = cor(nnRPLogPredict$net.result, testToyNorm$Y)
#view the correlation coefficient
nnRPLogCor
#train neuralnet using rprop+ algorithm and tanh activation function
nnRPTanh = neuralnet(Y ~., trainToyNorm, algorithm = "rprop+", hidden = 1, linear.output = FALSE, act.fct = "tanh", stepmax = 1e7)
#view neuralnet using rprop+ algorithm and tanh activation function
nnRPTanh
#use neural net to predict with test data
nnRPTanhPredict = neuralnet::compute(nnRPTanh, testToyNorm)
#compute the correlation coefficient
nnRPTanhCor = cor(nnRPTanhPredict$net.result, testToyNorm$Y)
#view the correlation coefficient
nnRPTanhCor
#train neuralnet using rprop+ algorithm and default(sigmoid) activation function
nnRPSig = neuralnet(Y ~., trainToyNorm, algorithm = "rprop+", hidden = 1, linear.output = TRUE, stepmax = 1e7)
#view neuralnet using rprop+ algorithm and default(sigmoid) activation function
nnRPSig
#use neural net to predict with test data
nnRPSigPredict = neuralnet::compute(nnRPSig, testToyNorm)
#compute the correlation coefficient
nnRPSigCor = cor(nnRPSigPredict$net.result, testToyNorm$Y)
#view the correlation coefficient
nnRPSigCor

#set seed for reproducibility
set.seed(430)
#train neuralnet using rprop- algorithm and logistic activation function
nnRMLog = neuralnet(Y ~., trainToyNorm, algorithm = "rprop-", hidden = 1, linear.output = FALSE, act.fct = "logistic", stepmax = 1e7)
#view neuralnet using rprop- algorithm and logistic activation function
nnRMLog
#use neural net to predict with test data
nnRMLogPredict = neuralnet::compute(nnRMLog, testToyNorm)
#compute the correlation coefficient
nnRMLogCor = cor(nnRMLogPredict$net.result, testToyNorm$Y)
#view the correlation coefficient
nnRMLogCor
#train neuralnet using rprop- algorithm and tanh activation function
nnRMTanh = neuralnet(Y ~., trainToyNorm, algorithm = "rprop-", hidden = 1, linear.output = FALSE, act.fct = "tanh", stepmax = 1e7)
#view neuralnet using rprop- algorithm and tanh activation function
nnRMTanh
#use neural net to predict with test data
nnRMTanhPredict = neuralnet::compute(nnRMTanh, testToyNorm)
#compute the correlation coefficient
nnRMTanhCor = cor(nnRMTanhPredict$net.result, testToyNorm$Y)
#view the correlation coefficient
nnRMTanhCor
#train neuralnet using rprop- algorithm and default(sigmoid) activation function
nnRMSig = neuralnet(Y ~., trainToyNorm, algorithm = "rprop-", hidden = 1, linear.output = TRUE, stepmax = 1e7)
#view neuralnet using rprop- algorithm and default(sigmoid) activation function
nnRMSig
#use neural net to predict with test data
nnRMSigPredict = neuralnet::compute(nnRMSig, testToyNorm)
#compute the correlation coefficient
nnRMSigCor = cor(nnRMSigPredict$net.result, testToyNorm$Y)
#view the correlation coefficient
nnRMSigCor
#close plotting device
dev.off()
#plot neuralnet that uses rprop+ with the sigmoid activation function
plot(nnRPSig, cex.axis = 0.2)


#2b
#set seed for reproducibility
set.seed(430)
#initialise vector to store result results matrices
resultMatrix = vector(mode="list", length = 7)
#initialise vector to store correlation coefficients
corResults = vector(mode = "list", length = 7)
#initialise vector to store rounded correlation coefficients
roundCorResults = vector(mode = "list", length = 7)
#initialise vector to use as x axis labels
xVec = c(2:8)
#for loop to iterate through adding one extra hidden node at a time
for (i in 2:8){
  nn = neuralnet(Y ~., trainToyNorm, hidden = i, algorithm = "rprop+", linear.output = TRUE, stepmax = 1e7)
  resultMatrix[[i-1]] = nn$result.matrix
  nnPredict = neuralnet::compute(nn, testToyNorm[,1:10])
  corResults[[i-1]] = cor(nnPredict$net.result, testToyNorm$Y)
  print(resultMatrix[[i-1]])
}
#view correlation coefficients
corResults
#view result matrices
resultMatrix
#for loop to iterate through correlation coefficients to round down to 3 decimal places
for (j in 1:7){
  roundCorResults[[j]] = round(corResults[[j]], digits=3)
}
#plot number of hidden units vs correlation coefficient
plot(xVec, corResults, xlab = "Number of Hidden Units", ylab = "Correlation Coefficient", main = "Correlation Coefficient of Test Data Against Number of Hidden Units", pch=16, col= "red")
#add labels to data points
text(xVec, corResults, labels=roundCorResults, pos=1, col = "blue")
#add grid lines to plot
grid(col = "lightgray")
#plot the neuralnet with 8 hidden nodes
plot(nn)
#use the compute function in order to access the net.result matrix
predictNN = neuralnet::compute(nn, testToyNorm[,1:10])
#unnormalize the net.result matrix
predictNN = (predictNN$net.result * (max(Toy$Y) - min(Toy$Y))) + min(Toy$Y)
#plot the predicted vs actual Y values
plot(testToy$Y, predictNN, col ="blue", pch=16, ylab = "Predicted Y", xlab = "Actual Y", main = "Unnormalized Predicted Vs Actual Response Variable (Y)")
#add regression line to the plot
abline(0,1)
#calculate the number of Y values in the unnormalized training data that are less than 5
sum(trainingToy$Y < 5)
#calculate the number of Y values in the unnormalized training data that are greater than 5
sum(trainingToy$Y > 5)

#2c
#set seed for reproducibility
set.seed(430)
#initialise vector for training splits
trainVec = c(0.2, 0.4, 0.6, 0.8)
#initialise vector to store result results matrices
resultMatrixC = vector(mode="list", length = 4)
#initialise vector to store correlation coefficients
corResultsC = vector(mode = "list", length = 4)
#initialise vector to store rounded correlation coefficients
roundCorResultsC = vector(mode = "list", length = 4)
#initialise q variable to 0
q = 0
#for loop to iterate through varying levels of train:test splits and create neural nets with each
for (k in trainVec){
  q = q+1
  sample = sample.split(Toy$X, SplitRatio = k) 
  trainingToyC = subset(Toy, sample == TRUE) 
  testToyC = subset(Toy, sample == FALSE)
  print(nrow(trainingToyC))
  print(nrow(testToyC))
  trainToyNormC = as.data.frame(lapply(trainingToyC, normalize))
  testToyNormC = as.data.frame(lapply(testToyC, normalize))
  nnC = neuralnet(Y ~., trainToyNormC, hidden = 8, linear.output = TRUE, algorithm = "rprop+", stepmax = 1e7)
  resultMatrixC[[q]] = nnC$result.matrix
  print(resultMatrixC[[q]])
  nnPredictC = neuralnet::compute(nnC, testToyNormC[,1:10])
  corResultsC[[q]] = cor(nnPredictC$net.result, testToyNormC$Y)
  roundCorResultsC[[q]] = round(corResultsC[[q]], digits=3)
  
}
#view correlation coefficients
corResultsC
#view result matrices
resultMatrixC
#close plotting device
dev.off()
#initialise vector with labels for x axis
testVec = c(20, 40, 60, 80)
#plot % of data used for training vs correlation coefficient
plot(testVec, corResultsC, xlab = "Percentage of Data Used for Training", ylab = "Correlation Coefficient", main = "Correlation Coefficient Against % of Data Used for Training", pch=16, col= "red")
#add labels to data points
text(testVec, corResultsC, labels=roundCorResultsC, pos=1, col = "blue")
#add grid lines to plot
grid(col = "lightgray")


