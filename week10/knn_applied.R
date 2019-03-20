# lab 4.6.6 - applying kNN
df = Caravan
names(df) = tolower(names(df))
X = scale(df[,-86])

test = 1:1000
Xtrain = X[-test,]
Xtest = X[test,]
ytrain = df$purchase[-test]
ytest = df$purchase[test]

set.seed(1)
knn.pred = knn(Xtrain, Xtest, ytrain, k=1)
mean(ytest != knn.pred) # error rate
mean(ytest != 'No') # trivial error rate if we just guess nos

