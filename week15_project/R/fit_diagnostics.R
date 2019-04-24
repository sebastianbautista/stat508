# fit statistics for fitted models
# input: data, fitted model
# output: .txt file containing confusion matrix and calculated rates

library(caret)

set.seed(202)
df = read.csv('../data/OnlineNewsPopularity.csv')

# create variables for classification, drop unused variables
df$y = as.factor(ifelse(df$shares >= quantile(df$shares, 0.75), 1, 0))
df[,c('shares', 'url', 'timedelta', 'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04')] = list(NULL)
levels(df$y) = c('low', 'high')

# X/y, train-test split
train = createDataPartition(df$y, p=0.66, list=F)
df.train = df[train,]
df.test = df[-train,]

# define func to calculate rates from confusion matrix
calc_rates = function(yhat, ytest){
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

# define func to load model, predict, and save confusion matrix/rates
model_diagnostics = function(filename){
  path = sprintf('../models/%s.rds', filename)
  model = readRDS(path)
  yhat = predict(model, df.test, type='raw')
  calc_rates(yhat, df.test$y)
  outfile = sprintf('../models/%s_cm.txt', filename)
  capture.output(calc_rates(yhat, df.test$y), file=outfile)
}

model_diagnostics('glmnet')

