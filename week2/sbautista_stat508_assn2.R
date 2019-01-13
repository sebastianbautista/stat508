"""
Refer to the Carseats dataset in the ISLR library.

Before you embark on your analysis, please review ISLR Section 3.6
(Links to an external site.)Links to an external site.
and carefully go over the annotated R lab that I have created
to guide you through this week's concepts.

Examine the model that predicts Sales based on all the other
variables in the dataset.

carseats.lm = lm(Sales~ ., data=Carseats)

Conduct a suite of regression diagnostics for this model,
explaining your choices.

If there is evidence of multicollinearity in the data,
refit the model using an appropriate subset of the covariates.
If transforming any of the variables in the model
(including the response variable) impacts your diagnostics,
apply a judiciously chosen transformation.

Write up your findings in a report, following the usual
guidelines, and be sure to include your code in an Appendix
so that it can be reproduced.
"""

library(ISLR)
library(ggplot2)
library(car)
library(gvlma)

df = Carseats

# Regression and results
reg = lm(Sales ~ ., data=df)
summary(reg)

# More verbose results from gvlma
summary(gvlma(reg, alphalevel=0.01))

# Plotting fitted vs studentized residuals
qplot(fitted(reg), rstudent(reg))
# No visual pattern in the residuals

# Histogram of residuals
qplot(rstudent(reg))
qqPlot(reg, main='QQ plot') # also looks for normality
# Roughly normal with mean 0

# Trying out functions from `car` library
outlierTest(reg)
# No outliers detected

# Non constant variance test
# Breusch-Pagan test for heterosked
ncvTest(reg)
# p - 0.77 means we fail to reject the null of homoskedasticity

# VIF for multicollinearity
# ISLR: VIF > 5 to 10 indicates problems
vif(reg)
# All of them are around 1-2, no evidence of multicollinearity

# Durbin-Watson test for autocorrelation in residuals
dwt(reg)
# High p-value = no evidence of serial correlation
