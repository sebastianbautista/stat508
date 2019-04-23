# training classification models on the Online News data
# threshold at 50% (median)
# input: OnlineNewsPopularity.csv
# output: saved models and text files containing diagnostics

library(caret)
library(glmnet)
library(randomForest)
library(kernlab)
library(e1071)
library(doParallel)

### SETUP ###
# for parallelization
cl = makePSOCKcluster(8)
registerDoParallel(cl)

set.seed(202)
df = read.csv('../data/OnlineNewsPopularity.csv')

# create variables for classification, drop unused variables
df$shares50 = as.factor(ifelse(df$shares > median(df$shares), 1, 0))
df[,c('shares', 'url', 'timedelta', 'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04')] = list(NULL)

# X/y, train-test split
train = createDataPartition(df$shares50, p=0.66, list=F)

X = df[,-(ncol(df))]
y = df$shares50

Xtrain = X[train,]
Xtest = X[-train,]
Xtrain.sc = scale(Xtrain)
Xtest.sc = scale(Xtest)

ytrain = y[train]
ytest = y[-train]

# set how cross-validation should be done (10 fold)
trControl = trainControl(method='cv', number=10)

# define func to pull out best caret fit + statistics
get_best = function(caret.fit){
  best = which(rownames(caret.fit$results) == rownames(caret.fit$bestTune))
  best.model = caret.fit$results[best,]
  rownames(best.model) = NULL
  best.model
}

# define func to calculate rates from confusion matrix
calc_rates = function(model){
  yhat = predict(model, Xtest)
  
  cm = table(yhat, ytest)
  
  accuracy = (cm[2,2]+cm[1,1])/sum(cm)
  error.rate = 1 - accuracy
  recall = cm[2,2]/sum(cm[,2])
  specificity = cm[1,1]/sum(cm[,1])
  precision = cm[2,2]/sum(cm[2,])
  
  cat('\n')
  print(cm)
  
  cat('\nAccuracy: ', accuracy)
  cat('\nError rate: ', error.rate)
  cat('\nRecall: ', recall)
  cat('\nSpecificity: ', specificity)
  cat('\nPrecision: ', precision)
  cat('\n')  
}

calc_rates_sc = function(model){
  yhat = predict(model, Xtest.sc)
  
  cm = table(yhat, ytest)
  
  accuracy = (cm[2,2]+cm[1,1])/sum(cm)
  error.rate = 1 - accuracy
  recall = cm[2,2]/sum(cm[,2])
  specificity = cm[1,1]/sum(cm[,1])
  precision = cm[2,2]/sum(cm[2,])
  
  cat('\n')
  print(cm)
  
  cat('\nAccuracy: ', accuracy)
  cat('\nError rate: ', error.rate)
  cat('\nRecall: ', recall)
  cat('\nSpecificity: ', specificity)
  cat('\nPrecision: ', precision)
  cat('\n')  
}

# define func to save the model and fit diagnostics
save_model = function(model, name){
  file.rds = sprintf('../models/%s.rds', name)
  file.txt = sprintf('../models/%s.txt', name)
  saveRDS(model, file=file.rds)
  capture.output(get_best(model), file=file.txt)
  capture.output(varImp(model), file=file.txt, append=T)
  capture.output(calc_rates(model), file=file.txt, append=T)
  capture.output(model, file=file.txt, append=T)
}

save_model_sc = function(model, name){
  file.rds = sprintf('../models/%s.rds', name)
  file.txt = sprintf('../models/%s.txt', name)
  saveRDS(model, file=file.rds)
  capture.output(get_best(model), file=file.txt)
  capture.output(varImp(model), file=file.txt, append=T)
  capture.output(calc_rates_sc(model), file=file.txt, append=T)
  capture.output(model, file=file.txt, append=T)
}

### ELASTIC NET ###
# first trying glmnet (elastic net), getting variable importances,
# then printing model object
model.glmnet = train(Xtrain.sc, ytrain,
                     method='glmnet',
                     trControl=trControl,
                     tuneLength=50,
                     metric='Accuracy',
                     maximize=T)
save_model_sc(model.glmnet, name='glmnet50')

### KNN ###
model.knn = train(Xtrain.sc, ytrain,
                     method='knn',
                     trControl=trControl,
                     tuneLength=50,
                     metric='Accuracy',
                     maximize=T)
save_model_sc(model.knn, name='knn50')

### RANDOM FOREST ###
model.rf = train(Xtrain, ytrain,
                     method='rf',
                     trControl=trControl,
                     tuneLength=50,
                     metric='Accuracy',
                     maximize=T)
save_model(model.rf, name='rf50')

stopCluster(cl)
