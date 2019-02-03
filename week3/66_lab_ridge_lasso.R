# 6.6 Lab 2: Ridge Regression and the Lasso

library(ISLR)
library(glmnet)

# Remove obs with missing values
df = na.omit(Hitters)

# Produces a matrix corresponding to 19 predictors and automatically dummifies categorical vars
# glmnet() can only take numerical, quantitative inputs
x = model.matrix(Salary ~ ., df)[,-1]
y = df$Salary

# 6.6.1 Ridge Regression
# in glmnet(), alpha=0 means ridge and alpha=1 means lasso
# lambda is 100 numbers between 10^10 and 10^2
grid = 10^seq(10, -2, length=100)
ridge.mod = glmnet(x, y, alpha=0, lambda=grid)

# coefficients are a 20*100 matrix
# 19 predictors + intercept * 100 lambdas
dim(coef(ridge.mod))

# large lambda should mean small coefficient estimates 
# value of lambda
ridge.mod$lambda[50]
# coefficients of the model with that lambda
coef(ridge.mod)[,50]
# l2 norm
sqrt(sum(coef(ridge.mod)[-1,50]^2))

# smaller lambda here, larger l2 norm and coefficients
ridge.mod$lambda[60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))

# we can use predict() to find the coefficients for a new lambda, 50
predict(ridge.mod, s=50, type='coefficients')[1:20,]

# now we do a train-test split
set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
ytest = y[test]

# next we fit and predict a ridge model using lambda=4
# we get predictions by using the `newx` argument
ridge.mod = glmnet(x[train,], y[train], alpha=0, lambda=grid, thresh=1e-12)
ridge.pred = predict(ridge.mod, s=4, newx=x[test,])
mean((ridge.pred - ytest)^2)

# we can compute test set MSE for an intercept-only model by using means
mean((mean(y[train]) - ytest)^2)

# we can get a very similar result by using ridge with a high lambda
ridge.pred = predict(ridge.mod, s=1e10, newx=x[test,])
mean((ridge.pred - ytest)^2)

# next we can check if OLS is better
# equivalent to ridge with lambda=0
ridge.pred = predict(ridge.mod, s=0, newx=x[test,], exact=T)
mean((ridge.pred - ytest)^2)

#####
# in general, we'll want to use cross validation to settle on lambda
set.seed(1)
cv.out = cv.glmnet(x[train,], y[train], alpha=0)
plot(cv.out)
bestlam = cv.out$lambda.min
bestlam

# what's the test MSE for this value of lambda?
ridge.pred = predict(ridge.mod, s=bestlam, newx=x[test,])
mean((ridge.pred - ytest)^2)

# even better!
# finally we refit on the full data set using bestlam and look at coefficients
out = glmnet(x, y, alpha=0)
predict(out, type='coefficients', s=bestlam)[1:20,]

#####
# 6.6.2 The Lasso


