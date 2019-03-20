# lab 4.6.5 - kNN

library(ISLR)
library(MASS)
library(class)
df = Smarket
names(df) = tolower(names(df))

train = (df$year < 2005)
df.test = df[!train,]

Xtrain = cbind(df$lag1, df$lag2)[train,]
Xtest= cbind(df$lag1, df$lag2)[!train,]
ytrain = df$direction[train]
ytest = df$direction[!train]

# k=1. too flexible
set.seed(1)
knn.pred = knn(Xtrain, Xtest, ytrain, k=1)
table(knn.pred, ytest)
mean(knn.pred == ytest)

# k=3, not great, qda is best for this data
knn.pred = knn(Xtrain, Xtest, ytrain, k=3)
table(knn.pred, ytest)
mean(knn.pred == ytest)