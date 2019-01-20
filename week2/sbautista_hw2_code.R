# R code for assignment 2

library(ISLR)
library(ggplot2)
library(corrplot)
library(car)

# Read in data, check structure
df = Carseats
str(df)

# Correlation table
num = sapply(df, is.numeric)
corrplot(cor(df[,num]), method='number', type='upper', bg='black')

# Base model
reg = lm(Sales ~ ., data=df)
summary(reg)

# Diagnostics, plots, etc.
round(confint(reg), 2)
plot(reg, which=1)
plot(reg, which=2)
round(vif(reg), 2)
outlierTest(reg, cutoff=Inf, n.max=5)
ncvTest(reg)

# First attempt
reg0 = lm(Sales ~ . + Income:Advertising, data=df)
summary(reg0)

# Second attempt after removing outliers
df1 = df[-c(358, 298, 357, 208, 16),]
reg1 = lm(Sales ~ . + Income:Advertising, data=df1)
summary(reg1)

# Third attempt
reg2 = lm(Sales ~ . + Income:Advertising + Income:Price, data=df1)
summary(reg2)