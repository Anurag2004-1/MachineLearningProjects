# importing liabraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read dataset
dataset =pd.read_excel(r"C:\Users\ASUS\OneDrive\Desktop\dataset\practice_dataset\Folds5x2_pp.xlsx")


# find out the best feature from dataset that has great result on our dependent variable in this case it is AT that is tempreture
# {# Calculate the correlation matrix
# correlation_matrix = dataset.corr()
#
# # Extract the correlation coefficients between each feature and the target variable
# correlation_with_target = correlation_matrix['PE'].abs()
#
# # Sort the correlations in descending order
# sorted_correlations = correlation_with_target.sort_values(ascending=False)
#
# print(sorted_correlations)}

x= dataset.iloc[2000:,0].values
y=dataset.iloc[2000:,-1].values

#splitting dataset into testset and traing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.30,random_state=42)

#reshaping x_train and x_test
x_train = np.array(x_train).reshape(-1, 1)
x_test= np.array(x_test).reshape(-1, 1)


#training dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting new result
print(regressor.predict([[15]]))
#expected output:463.26
#acutal output:464.484

#predicting test set
y_pred=regressor.predict(x_test)

#visulizing trsining set result
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("combined cycle power plant(training_set) ")
plt.xlabel("temprature")
plt.ylabel("PE (Net Hourly Electrical Energy Output)")
plt.show()

#visualize test set result
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("combined cycle power plant(test_set)")
plt.xlabel("temprature")
plt.ylabel("PE (Net Hourly Electrical Energy Output)")
plt.show()


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Importing SVR from scikit-learn
from sklearn.svm import SVR

# Creating and training an SVR model
svr_regressor = SVR(kernel='linear')  # You can explore different kernel options: 'linear', 'poly', 'rbf', etc.
svr_regressor.fit(x_train, y_train)

# Predicting test set
y_pred_svr = svr_regressor.predict(x_test)




