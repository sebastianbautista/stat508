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

#######################################
df = Carseats

# Shape - 400 by 11
dim(df)

# Feature names
names(df)

# Summary stats
summary(df)

# Correlation matrix
nums = unlist(lapply(df, is.numeric))
df_num = df[, nums]
cor(df_num)

#######################################

# Regression and results
reg = lm(Sales ~ ., data=df) 
summary(reg)

# More verbose results from gvlma
# TODO: add interpretation of parts
summary(gvlma(reg, alphalevel=0.01))

# Plotting fitted vs studentized residuals
qplot(fitted(reg), rstudent(reg))
# No visual pattern in the residuals

# Histogram of residuals 
qplot(rstudent(reg))
# Roughly normal with mean 0

### Trying out functions from `car` library 
# https://www.statmethods.net/stats/rdiagnostics.html
# no studentized residuals with Bonferonni p < .05
outlierTest(reg) 
# No outliers detected

# quantile-quantile plot looks pretty decent
qqPlot(reg, main='QQ plot')
# TODO: add exact interpretation. Normality again?

# leverage - not sure how to interpret these
# TODO: clarify
leveragePlots(reg)

# Non constant variance test
# Breusch-Pagan test for heterosked
ncvTest(reg)
# p - 0.77 means we fail to reject the null of homoskedasticity

# Doesn't look like fitted vs resid but what is it?
# Studentized residuals vs fitted?
# TODO: find out what this is
spreadLevelPlot(reg)

# VIF for multicollinearity
# ISLR: VIF > 5 to 10 indicates problems
vif(reg)
sqrt(vif(reg)) > 2 # basically checks if any of them are greater than 4?
# All of them are around 1-2, no evidence of multicollinearity
# TODO: see if removing advertising or US helps?? probably not

# Component residual plots
# TODO: interpret, maybe not useful
crPlots(reg)

# Durbin-Watson test for autocorrelation in residuals
dwt(reg)
# High p-value = no evidence of serial correlation
