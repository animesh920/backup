# VIDEO 4

# Read in data
wine = read.csv("wine.csv")
str(wine)
summary(wine)

# Linear Regression (one variable)
model1 = lm(Price ~ AGST, data=wine)
summary(model1)

# Sum of Squared Errors
model1$residuals
SSE = sum(model1$residuals^2)
SSE

# Linear Regression (two variables)
model2 = lm(Price ~ AGST + HarvestRain, data=wine)
summary(model2)

# Sum of Squared Errors
SSE = sum(model2$residuals^2)
SSE

# Linear Regression (all variables)
model3 = lm(Price ~ AGST + HarvestRain + WinterRain + Age + FrancePop, data=wine)
summary(model3)

# Sum of Squared Errors
SSE = sum(model3$residuals^2)
SSE


# VIDEO 5

# Remove FrancePop
model4 = lm(Price ~ AGST + HarvestRain + WinterRain + Age, data=wine)
summary(model4)


# VIDEO 6

# Correlations
cor(wine$WinterRain, wine$Price)
cor(wine$Age, wine$FrancePop)
cor(wine)

#multi collinearity is the problem when the two independent variable are highly correlated. such dependence increa
#the R2 without giving any extra information. keep only one of the collinear variable in the model. this also means
#that i have to keep the vairable that makes intuitive sense to keep in the model.

# Remove Age and FrancePop

#the age and the france pop were removed because the summary shows that they are not that well related to the dep
#variable

model5 = lm(Price ~ AGST + HarvestRain + WinterRain, data=wine) #AGST is kept because it makes sense to keep it.
summary(model5)

#high correlation can even cause a singnificant variable have one intuitive sign. 

#SST is the base model, the base model predicts the average value of y for all values of x. so the R2 measures how
#good my model is from the base model      1-(SSE/SST)


#on adding more variable to the model the model R2 always increases and it can never decrease. but the test set
#R2 can decrease on adding more variable and even be negative. this means that adding more variable in the model
#always is not a good idea.


#the CV method is used to decide which model to use


# VIDEO 7

# Read in test set
wineTest = read.csv("wine_test.csv")
str(wineTest)

#one way to deal with the missing observation is to remove the observation altogether, another way to deal with the
#missing observation is to use imputation

# Make test set predictions
predictTest = predict(model4, newdata=wineTest)
predictTest

# Compute R-squared
SSE = sum((wineTest$Price - predictTest)^2)
SST = sum((wineTest$Price - mean(wine$Price))^2)
1 - SSE/SST

