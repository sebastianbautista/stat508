# training regression models on the Online News data
# input: OnlineNewsPopularity.csv
# output: saved models and text files containing training diagnostics
## note that this script only trains and saves the models
## TODO: write another script that loads the models and provides test fit stats

library(caret)
library(glmnet)
library(randomForest)
library(kernlab)
library(doParallel)

### SETUP ###
# for parallelization
cl = makePSOCKcluster(5)
registerDoParallel(cl)

set.seed(202)
df = read.csv('../data/OnlineNewsPopularity.csv')

# drop unused variables
df[,c('url', 'timedelta', 'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04')] = list(NULL)

# X/y, train-test split
train = createDataPartition(df$shares, p=0.66, list=F)

X = df[,-(ncol(df))]
y = df$shares
y.log = log(df$shares)

Xtrain = X[train,]
Xtest = X[-train,]
Xtrain.sc = scale(Xtrain)
Xtest.sc = scale(Xtest)
ytrain = y[train]
ytest = y[-train]
ytrain.log = y.log[train]
ytest.log = y.log[-train]

# set how cross-validation should be done (10 fold)
trControl = trainControl(method='cv', number=10)

# define func to pull out best caret fit + statistics
get_best = function(caret.fit){
  best = which(rownames(caret.fit$results) == rownames(caret.fit$bestTune))
  best.model = caret.fit$results[best,]
  rownames(best.model) = NULL
  best.model
}

# define func to save the model and training fit diagnostics
save_model = function(model, name){
  file.rds = sprintf('../models/%s.rds', name)
  file.txt = sprintf('../models/%s.txt', name)
  saveRDS(model, file=file.rds)
  capture.output(get_best(model), file=file.txt)
  capture.output(varImp(model.glmnet), file=file.txt, append=T)
  capture.output(model, file=file.txt, append=T)
  
}

### ELASTIC NET ###
# first trying glmnet (elastic net), getting variable importances,
# then printing model object
model.glmnet = train(Xtrain.sc, ytrain,
                     method='glmnet',
                     trControl=trControl,
                     tuneLength=50,
                     metric='Rsquared',
                     maximize=T)
varImp(model.glmnet)
get_best(model.glmnet)
save_model(model.glmnet, 'glmnet')

### ELASTIC NET, LOG ###
# first trying glmnet (elastic net), getting variable importances,
# then printing model object
model.glmnet.log = train(Xtrain.sc, ytrain.log,
                     method='glmnet',
                     trControl=trControl,
                     tuneLength=50,
                     metric='Rsquared',
                     maximize=T)
varImp(model.glmnet.log)
get_best(model.glmnet.log)
save_model(model.glmnet.log, 'glmnetlog')

stopCluster(cl)
