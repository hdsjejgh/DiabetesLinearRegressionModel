print("Importing Dataset")
from sklearn import datasets, linear_model #imports datasets and linear regressions model
from sklearn.model_selection import train_test_split #lets you train and test
from sklearn.metrics import mean_squared_error, r2_score #checks error
from time import sleep
import seaborn as sns #plot the equation thing
import matplotlib.pyplot as plt #plot the equation thing

diabetes = datasets.load_diabetes()
x = diabetes.data #10 different factors
y = diabetes.target #progression of diabetes
print("Imported\n")
sleep(0.5)
#print(diabetes['DESCR'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1) #10% of data is reserved for testing
model = linear_model.LinearRegression() #makes the model itself

model.fit(x_train, y_train) #trains the model
y_pred = model.predict(x_test) #tests the model
slopes = model.coef_ #all coefficients for every variable
intercept = model.intercept_ #y intercept
error = mean_squared_error(y_test,y_pred) #honestly no clue what mean squared error is but it sounds important
r2 = r2_score(y_test, y_pred) #r^2 is basically accuracy score
VARIABLE_LIST= list('abcdefghij') #list of variable
equation = "y = "+(' + '.join(tuple(f"({slopes[idx]}\033[1m{variable}\033[0m)" for idx, variable in enumerate(VARIABLE_LIST))))+f" + ({intercept})" #makes the equation
print(f"\033[1mCoefficients:\033[0m {slopes}\n\033[1mY-Intercept:\033[0m {intercept}\n\033[1mMean Squared Error:\033[0m {error}\n\033[1mAccuracy:\033[0m {100*r2:.3f}%") #prints all the data
print(f"\n\033[1mEquation: \033[0m{equation}") #prints equation

sns.scatterplot(x=y_test,y=y_pred,alpha=0.5)
plt.show()#shows scatter plot (the closer x is to y is the closer the prediction was)