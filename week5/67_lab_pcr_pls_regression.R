# 6.7 Lab 3: PCR and PLS Regression

### 6.7.1: Principal Components Regression

library(ISLR)
library(pls)

# Remove missings
df = na.omit(Hitters)

x = model.matrix(Salary ~ ., df)[,-1]
y = df$Salary

# train test split
set.seed(1)
train = sample(1:nrow(df), nrow(df)/2)
test = (-train)

Xtrain = x[train,]
Xtest = x[test,]
ytrain = y[train]
ytest = y[test]

# Fit PCR, making sure to scale and use 10-fold CV
set.seed(2)
pcr.fit = pcr(Salary ~ ., data=df, scale=T, validation='CV')

# CV score reported is RMSE
# % variance explained shows the amount of information captured using M principal components
summary(pcr.fit)

# plot CV scores (MSE)
# M=16 components are chosen, but M=1 is also pretty good!
validationplot(pcr.fit, val.type='MSEP')

# PCR on training data and evaluate test set
# the lowest CV error occurs when M=7 components are used
set.seed(1)
pcr.fit = pcr(Salary ~ ., data=df, subset=train, scale=T, validation='CV')
validationplot(pcr.fit, val.type='MSEP')

# compute test MSE with M=7
pcr.pred = predict(pcr.fit, Xtest, ncomp=7)
mean((pcr.pred - ytest)^2)
# 96556, comparable to ridge/lasso, but is harder to interpret because it 
# doesn't perform variable selection or directly produce coefficient estimates

# refit PCR on the full data set using M=7
pcr.fit.full = pcr(y~x, scale=T, ncomp=7)
summary(pcr.fit.full)

### 6.7.2: Partial Least Squares

set.seed(1)
pls.fit = plsr(Salary ~ ., data=df, subset=train, scale=T, validation='CV')

# lowest CV error is at M=2 pls directions
summary(pls.fit)
validationplot(pls.fit, val.type='MSEP')

# evaluate corresponding test MSE
pls.pred = predict(pls.fit, Xtest, ncomp=2)
mean((pls.pred - ytest)^2)

# finally refit using the full data set
pls.fit.full = plsr(Salary ~ ., data=df, scale=T, ncomp=2)
summary(pls.fit.full)
# the percentage of variance in Salary that this explains, 46.40%, is almost as much as
# the M=7 PCR fit, 46.69%.
# this is because PCR only attempts to maximize the amount of variance
# explained in the predictors, while
# PLS searches for directions that explain variance in both the predictors and response
