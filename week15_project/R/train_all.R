# training classification models on the Online News data
# input: OnlineNewsPopularity.csv
# output: saved models and text files containing diagnostics

library(caret)
library(doParallel)
library(ROCR)
library(glmnet)
library(kernlab)
library(e1071)
library(fastAdaboost)

### SETUP ###
# for parallelization
cl = makePSOCKcluster(8)
registerDoParallel(cl)

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

# set how cross-validation should be done
k = 3
index = createFolds(factor(df.train$y), k=k, returnTrain=T)
trControl = trainControl(method='cv', 
                         index=index,
                         number=k, 
                         classProbs=T,
                         summaryFunction=twoClassSummary)

# define func to pull out best caret fit + statistics
get_best = function(caret.fit){
  best = which(rownames(caret.fit$results) == rownames(caret.fit$bestTune))
  best.model = caret.fit$results[best,]
  rownames(best.model) = NULL
  best.model
}

# define func to report test auc
report_auc = function(model){
  yhat = predict(model, df.test)
  predob = prediction(as.numeric(yhat), as.numeric(df.test$y))
  auc = performance(predob, measure='auc')
  cat('\nTest AUC: ', auc@y.values[[1]])
  cat('\n')  
}

# define func to save the model and fit diagnostics
save_model = function(model, name){
  file.rds = sprintf('../models/%s.rds', name)
  file.txt = sprintf('../models/%s.txt', name)
  saveRDS(model, file=file.rds)
  capture.output(get_best(model), file=file.txt)
  capture.output(varImp(model), file=file.txt, append=T)
  capture.output(report_auc(model), file=file.txt, append=T)
  capture.output(model, file=file.txt, append=T)
}

# ### ELASTIC NET ###
# # first trying glmnet (elastic net), getting variable importances,
# # then printing model object
# model.glmnet = train(y~.,
#                      data=df.train,
#                      method='glmnet',
#                      trControl=trControl,
#                      preProc=c('center', 'scale'),
#                      tuneLength=10,
#                      metric='ROC',
#                      maximize=T)
# save_model(model.glmnet, name='glmnet')

# ### ELASTIC NET, LOG ### problematic, won't see '.'
# model.glmnet.log = train(y~log(.),
#                      data=df.train,
#                      method='glmnet',
#                      trControl=trControl,
#                      preProc=c('center', 'scale'),
#                      tuneLength=10,
#                      metric='ROC',
#                      maximize=T)
# save_model(model.glmnet.log, name='glmnet_log')

# ### KNN ###
# model.knn = train(y~.,
#                   data=df.train,
#                   method='knn',
#                   trControl=trControl,
#                   preProc=c('center', 'scale'),
#                   tuneLength=10,
#                   metric='ROC',
#                   maximize=T)
# save_model(model.knn, name='knn')

### ADAPTIVE BOOSTING ###
model.ada = train(y~.,
                  data=df.train,
                  method='adaboost',
                  trControl=trControl,
                  tuneLength=10,
                  metric='ROC',
                  maximize=T)
save_model(model.ada, name='ada')

### SVM, linear ###
model.svml = train(y~.,
                   data=df.train,
                   method='svmLinear',
                   trControl=trControl,
                   tuneLength=5,
                   metric='ROC',
                   maximize=T)
save_model(model.svml, name='svml')

### SVM, poly ###
model.svmp = train(y~.,
                   data=df.train,
                   method='svmPoly',
                   trControl=trControl,
                   tuneLength=5,
                   metric='ROC',
                   maximize=T)
save_model(model.svmp, name='svmp')

### SVM, radial ###
model.svmr = train(y~.,
                   data=df.train,
                   method='svmRadial',
                   trControl=trControl,
                   tuneLength=5,
                   metric='ROC',
                   maximize=T)
save_model(model.svmr, name='svmr')

stopCluster(cl)
