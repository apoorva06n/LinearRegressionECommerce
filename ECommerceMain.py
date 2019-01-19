#import helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def main():
	#read the csv file with data
	customers = pd.read_csv('./Ecommerce Customers.csv')
	#print the dataset information
	customers.info()
	customers.describe()
	#print jointplot to compare the Time on Website and Yearly Amount Spent columns
	sns.jointplot(data=customers,x=customers['Time on Website'],y=customers['Yearly Amount Spent'])
	plt.show()
	#print jointplot to compare the Time on App and Yearly Amount Spent columns
	sns.jointplot(data=customers,x=customers['Time on App'],y=customers['Yearly Amount Spent'])
	plt.show()
	#print jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership
	sns.jointplot(data=customers,x=customers['Time on App'],y=customers['Length of Membership'],kind='hex')
	plt.show()
	#print pairplot to explore types of relationships across the entire data set
	sns.pairplot(customers)
	plt.show()
	print("Based off this plot Length of Membership looks to be the most correlated feature with Yearly Amount Spent")
	sns.lmplot(data=customers,x='Length of Membership',y='Yearly Amount Spent')
	plt.show()
	lm_model,x_test,y_test = train_data(customers)
	predictions = predict_testResults(lm_model,x_test,y_test)
	evaluate(y_test,predictions)
	X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
	cdf = pd.DataFrame(lm_model.coef_ ,X.columns,columns=['Coeffecient'])
	print(cdf)

def train_data(customers):
	y = customers['Yearly Amount Spent']
	X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
	X_train,x_test,Y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=101)
	lm = LinearRegression()
	lm.fit(X_train,Y_train)
	return lm,x_test,y_test

def predict_testResults(lm,x_test,y_test):
	predictions = lm.predict(x_test)
	plt.scatter(y_test,predictions)
	plt.xlabel("Y Test")
	plt.ylabel('Predicted Y')
	plt.show()
	return predictions

def evaluate(y_test,predictions):
	print("MAE :",metrics.mean_absolute_error(y_test,predictions))
	print("MSE :",metrics.mean_squared_error(y_test,predictions))
	print("RMSE :",np.sqrt(metrics.mean_squared_error(y_test,predictions)))
	sns.distplot((y_test - predictions ),bins=50)
	plt.show()
	

if __name__== "__main__":
  main()
