from matplotlib.colors import Colormap
from seaborn.utils import ci
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import power_transform
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

filevessel = 'vesseldata.csv'
fileenv = 'envdata.csv'
filefuel = 'fueldata.csv'

vesseldf = pd.read_csv(filevessel)
envdf = pd.read_csv(fileenv)
fueldf = pd.read_csv(filefuel)

vesseldf["Date"] = pd.to_datetime(vesseldf["Date"])
vesseldf["Time"] = pd.to_datetime(vesseldf["Time"])

envdf["Date"] = pd.to_datetime(envdf["Date"])
envdf["Time"] = pd.to_datetime(envdf["Time"])

# fueldf["Date"] = pd.to_datetime(fueldf["Date"])
# fueldf["Time"] = pd.to_datetime(fueldf["Time"])

# print(f"{vesseldf.info()}\n {envdf.info()}\n{fueldf.info()}")

CombinedMetrics = pd.DataFrame()
CombinedMetrics = pd.concat([vesseldf, envdf, fueldf], axis=1) #joined all datasources into one DF for easier manipulation





# Setting Output Variables 
yVar = pd.DataFrame()
yVar = (CombinedMetrics[['SOX', 'NOX', 'Viscosity_cst']])

# Setting Input Variables 
Xvar = CombinedMetrics.drop(columns=['SOX', 'NOX', 'Date', 'Time', 'Viscosity_cst'], axis=1)

# print(Xvar, yVar)

XandY = pd.concat([Xvar, yVar], axis=1)

# define min max scaler
scaler = MinMaxScaler()
# transform data
normData = pd.DataFrame(scaler.fit_transform(XandY), index=XandY.index, columns=XandY.columns)
# scaleddata = pd.DataFrame(scaled)
print(normData)



# fig = plt.figure(figsize=(10,7))
# fig.add_subplot(2,1,1)
# sns.displot(normData['Wave_height'])
# plt.tight_layout()
# plt.show()
# sns.violinplot(y = normData['Viscosity_cst'], x = normData['Wave_height'])

# sns.pairplot(normData, palette = 'magma')


# plt.tight_layout()
# plt.show()

Xfeat= normData.iloc[:,0:13]
ytarget = normData.iloc[:,-1]


# X_train, X_test, y_train, y_test= train_test_split(
#     Xfeat, ytarget, test_size=0.3, random_state=1)





rf_model= RandomForestClassifier( max_depth=12, n_estimators = 300)


rf_model.fit(Xfeat.astype(int), ytarget.astype(int))



# train_features = X_train.columns
# importances = rf_model.feature_importances_
# indices = np.argsort(importances)[-8:]  # top features
# plt.title("Feature Importance")
# plt.barh(range(len(indices)),
#          importances[indices], color='g', align='center')
# plt.yticks(range(len(indices)), [
#            train_features[i] for i in indices])
# plt.xlabel('Relative Importance')
# # plt.xlim(0, 0.6)
# plt.tight_layout()
# plt.show()



# use feature importance for feature selection

from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt


# from xgboost import XGBRegressor

X_train, X_test, y_train, y_test= train_test_split(
    Xfeat, ytarget, test_size=0.33, random_state=101)

#standardization scaler - fit&transform on train, fit only on test
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))


# Multiple Liner Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
#evaluate the model (intercept and slope)
print(regressor.intercept_)
print(regressor.coef_)
#predicting the test set result
y_pred = regressor.predict(X_test)
#put results as a DataFrame
coeff_df = pd.DataFrame(regressor.coef_, normData.drop(['Viscosity_cst', 'NOX', 'SOX'],axis =1).columns, columns=['Coefficient']) 
print(coeff_df)

# visualizing residuals
fig = plt.figure(figsize=(10,5))
residuals = (y_test- y_pred)
sns.distplot(residuals)

plt.tight_layout()
plt.show()

#compare actual output values with predicted values
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(10)
print(df1)
# evaluate the performance of the algorithm (MAE - MSE - RMSE)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))




    
# Plotting the actual vs predicted values
sns.lmplot(x='Actual', y='Predicted', data=df, fit_reg=False, size=7)
    
# Plotting the diagonal line
line_coords = np.arange(df.min().min(), df.max().max())
plt.plot(line_coords, line_coords,  # X and y points
            color='darkorange', linestyle='--')
plt.title('Actual vs. Predicted')
plt.tight_layout()
plt.show()





  
# plt.barh(boston.feature_names, xgb.feature_importances_)

# X = normData.iloc[:,0:13].astype(int)  #independent columns
# y = normData.iloc[:,-1].astype(int)   #viscosity
# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
# model = ExtraTreesClassifier()
# model.fit(X,y)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(5).plot(kind='barh')
# plt.title('Feature Importance')
# plt.tight_layout()
# plt.show()



# Creating a Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam



# plt.figure(figsize=(12,10))
# cor = CombinedMetrics.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.tight_layout()
# plt.show()

# fig = plt.figure(figsize=(16,5))
# fig.add_subplot(2,2,1)
# sns.scatterplot(normData['Wave_height'], normData['Viscosity_cst'])
# fig.add_subplot(2,2,2)
# sns.scatterplot(normData['F.O_Temp_Celcius'],normData['Viscosity_cst'])
# fig.add_subplot(2,2,3)
# sns.scatterplot(normData['L.O.Main_Temp_Celcius'],normData['Viscosity_cst'])
# fig.add_subplot(2,2,4)
# sns.scatterplot(normData['L.O.Main_Pressure_kgm3'],normData['Viscosity_cst'])
# plt.tight_layout()
# plt.show()

# fig = plt.figure(figsize=(15,7))
# fig.add_subplot(2,2,1)
# sns.countplot(normData['Wave_height'])
# fig.add_subplot(2,2,2)
# sns.countplot(normData['F.O_Temp_Celcius'])
# fig.add_subplot(2,2,3)
# sns.countplot((normData['L.O.Main_Temp_Celcius']))
# fig.add_subplot(2,2,4)
# sns.countplot(normData['L.O.Main_Pressure_kgm3'])
# plt.tight_layout()
# plt.show()



# ###############################################################################
# F    E    A    T    U    R    E   •••••  S    E    L   L   E   C   T   I   O   N

