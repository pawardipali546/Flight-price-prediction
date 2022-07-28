# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:05:22 2022

@author: Dipali.Badgujar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading dataset
train = pd.read_excel(r"C:\360 CLASSES\flight 1 project\Data_Train.xlsx")
pd.set_option('display.max_columns', None)
train.head()
train.describe()
train.info()
train.columns
train['Duration'].value_counts()
train.isnull().sum()
train.dropna(inplace= True)
train.isnull().sum()

## EDA
train['Journey_day'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.day
train['Journey_month'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.month
train.head()

# we can drop Date_of_journey column
train.drop(["Date_of_Journey"],axis=1,inplace= True)

# extracting hr, minute from dep_time
train['Dep_hour'] = pd.to_datetime(train["Dep_Time"]).dt.hour
train['Dep_min'] = pd.to_datetime(train["Dep_Time"]).dt.minute

# we can drop dep_time column
train.drop(["Dep_Time"],axis=1,inplace= True)
train.head()

# extracting hr, minute from arrival_time
train['Arrival_hour'] = pd.to_datetime(train["Arrival_Time"]).dt.hour
train['Arrival_min'] = pd.to_datetime(train["Arrival_Time"]).dt.minute

# we can drop dep_time column
train.drop(["Arrival_Time"],axis=1,inplace= True)
train.head()

# Assigning and converting Duration column into list
duration = list(train["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding duration_hours and duration_mins list to train dataframe
train["Duration_hours"]= duration_hours
train["Duration_mins"] = duration_mins

train.drop(["Duration"],axis=1, inplace=True)
train.head()

# for categorical data : we are using one hot encoding for nominal and label encoding for ordinal data
train["Airline"].value_counts()

# Airline vs Price
sns.catplot(y = "Price", x = "Airline", data = train.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()

# one hot encoding
Airline = train[["Airline"]]
Airline = pd.get_dummies(Airline,drop_first=True)
Airline.head()

train["Source"].value_counts()

# Source vs price
sns.catplot(y= "Price" , x="Source" , data = train.sort_values("Price", ascending = False), kind = "boxen", height = 4, aspect =3)
plt.show()

#one hot encosding
source = train[["Source"]]
source=pd.get_dummies(source, drop_first= True)
source.head()

train["Destination"].value_counts()

# destination vs price
sns.catplot(y="Price", x= "Destination", data= train.sort_values("Price", ascending = False), kind = "boxen", height = 4, aspect= 3)
plt.show()

# one hot encoding
destination= train[["Destination"]]
destination= pd.get_dummies(destination,drop_first= True)
destination.head()

train["Route"]

# Route and Total_Stops are related to each other, so route and additional info are dropped
train.drop(["Route","Additional_Info"], axis=1, inplace= True)
train.columns

train["Total_Stops"].value_counts()

# label encoding are used for total_stop column
train.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops" : 4}, inplace = True)
train.head()

# Concatenate dataframe --> train_data + Airline + Source + Destination
data_train = pd.concat([train, Airline, source, destination], axis = 1)
data_train.head()

data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
data_train.head()

data_train.shape

# test dataset
test = pd.read_excel(r"C:\360 CLASSES\flight 1 project\Test_set.xlsx")
pd.set_option('display.max_columns', None)
test.head()
test.describe()
test.info()
test.columns
test['Duration'].value_counts()
train.isnull().sum()

# EDA

# Date_of_Journey
test["Journey_day"] = pd.to_datetime(test.Date_of_Journey, format="%d/%m/%Y").dt.day
test["Journey_month"] = pd.to_datetime(test["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test["Dep_hour"] = pd.to_datetime(test["Dep_Time"]).dt.hour
test["Dep_min"] = pd.to_datetime(test["Dep_Time"]).dt.minute
test.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test["Arrival_hour"] = pd.to_datetime(test.Arrival_Time).dt.hour
test["Arrival_min"] = pd.to_datetime(test.Arrival_Time).dt.minute
test.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test["Duration_hours"] = duration_hours
test["Duration_mins"] = duration_mins
test.drop(["Duration"], axis = 1, inplace = True)
test.columns

# for categorical data : we are using one hot encoding for nominal and label encoding for ordinal data
test["Airline"].value_counts()

# one hot encoding
Airline = test[["Airline"]]
Airline = pd.get_dummies(Airline,drop_first=True)
Airline.head()

test["Source"].value_counts()

#one hot encosding
source = test[["Source"]]
source=pd.get_dummies(source, drop_first= True)
source.head()

test["Destination"].value_counts()

# one hot encoding
destination= test[["Destination"]]
destination= pd.get_dummies(destination,drop_first= True)
destination.head()

test["Route"]

# Route and Total_Stops are related to each other, so route and additional info are dropped
test.drop(["Route","Additional_Info"], axis=1, inplace= True)
test.columns

test["Total_Stops"].value_counts()

# label encoding are used for total_stop column
test.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops" : 4}, inplace = True)
test.head()

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test, Airline,source, destination], axis = 1)
data_test.head()
data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
data_test.head()
data_test.shape

# Feature Selection : Finding out the best feature which will contribute and have good relation with target variable. Following are some of the feature selection methods,
#**heatmap**
# *feature_importance_**
# **SelectKBest**

data_train.shape
data_train.columns

X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()

y = data_train.iloc[:, 1]
y.head()

# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (18,18))
sns.heatmap(train.corr(), annot = True, cmap = "RdYlGn")
plt.show()

# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)

print(selection.feature_importances_)

#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

# Fitting model using Random Forest
#Split dataset into train and test set in order to prediction w.r.t X_test
#If needed do scaling of data
#Scaling is not done in Random forest
#Import model
#Fit the data
#Predict w.r.t X_test
#In regression check RSME Score
#Plot graph

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)

y_pred = reg_rf.predict(X_test)

reg_rf.score(X_train, y_train)

reg_rf.score(X_test, y_test)

sns.distplot(y_test-y_pred)
plt.show()

plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# RMSE/(max(DV)-min(DV))

2090.5509/(max(y)-min(y))

metrics.r2_score(y_test, y_pred)

# Hyperparameter Tuning
#Choose following method for hyperparameter tuning
#RandomizedSearchCV --> Fast
#GridSearchCV
#Assign hyperparameters in form of dictionery
#Fit the model
#Check best paramters and best score

from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)

rf_random.best_params_

prediction = rf_random.predict(X_test)

plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()

plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# save the model for reuseing again
import pickle

# open a file, where you ant to store the data
file = open('flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(reg_rf, file)

model = open('flight_rf.pkl','rb')
forest = pickle.load(model)
y_prediction = forest.predict(X_test)
metrics.r2_score(y_test, y_prediction)



































