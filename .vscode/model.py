from matplotlib.colors import Colormap
from seaborn.utils import ci
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import PowerTransformer

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

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




###################################
# D  A  T  A  ***  A  C  Q  U  I  S  I  T  I  O  N

CombinedMetrics = pd.DataFrame()
CombinedMetrics = pd.concat([vesseldf, envdf, fueldf], axis=1) #joined all datasources into one DF for easier manipulation





# Setting Output Variables 
yVar = pd.DataFrame()
yVar = (CombinedMetrics[['SOX', 'NOX', 'Viscosity_cst']])

# Setting Input Variables 
Xvar = CombinedMetrics.drop(columns=['SOX', 'NOX', 'Date', 'Time', 'Viscosity_cst'], axis=1)

# print(Xvar, yVar)

XandY = pd.concat([Xvar, yVar], axis=1)







#######################################
# M  I  N - M  A  X   ***  S  C  A  L  A  R

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





X_train, X_test, y_train, y_test= train_test_split(
    Xfeat, ytarget, test_size=0.33, random_state=101)

#standardization scaler - fit&transform on train, fit only on test

s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))






rf_model= RandomForestClassifier( max_depth=12, n_estimators = 300)


rf_model.fit(X_train.astype(int), y_train.astype(int))




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

# visualizing residuals (distance between predictions and actual through regression line)
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




# # ###############################################################################
# # F    E    A    T    U    R    E   •••••  S    E    L   L   E   C   T   I   O   N

  
# # plt.barh(boston.feature_names, xgb.feature_importances_)

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


# # #####################################
# # N  E  U  R  A  L - N  E  T  W  O  R  K 

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam



# # having 13 neuron is based on the number of available features
# model = Sequential()
# model.add(Dense(13,activation='relu'))
# model.add(Dense(13,activation='relu'))
# model.add(Dense(13,activation='relu'))
# model.add(Dense(13,activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='Adam',loss='mse')

# model.fit(x=X_train,y=y_train,
#           validation_data=(X_test,y_test),
#           batch_size=128,epochs=400)
# model.summary()


# loss_df = pd.DataFrame(model.history.history)
# loss_df.plot(figsize=(12,8))

# y_pred = model.predict(X_test)
# from sklearn import metrics
# print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
# print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('VarScore:',metrics.explained_variance_score(y_test,y_pred))
# # Visualizing Our predictions
# fig = plt.figure(figsize=(10,5))
# plt.scatter(y_test,y_pred)
# # Perfect predictions
# plt.plot(y_test,y_test,'r')
# plt.tight_layout()
# plt.show()









# # visualizing residuals
# fig = plt.figure(figsize=(10,5))
# x_test = np.reshape(X_test(-1, 1))

# residuals = (y_test- y_pred)
# sns.distplot(residuals)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12,10))
# cor = CombinedMetrics.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.tight_layout()
# plt.show()


pt = PowerTransformer(method='yeo-johnson' )
dataTransform = pt.fit_transform(Xfeat)
# convert the array back to a dataframe
datasetYeo = pd.DataFrame(dataTransform, columns=['Load_pct', 'Engine-RPM', 'SOG', 'STW', 'DFT', 'L.O.Main_Pressure_kgm3',
                                                  'L.O.Main_Temp_Celcius', 'RWS', 'RWD' ,'Wave_height','Air_Temp', 'F.O_pressure','F.O_Temp_Celcius'  ])
# histograms of the variables
# plt.hist(datasetYeo)
# plt.tight_layout()
# plt.show()

print(datasetYeo.head(6))





fig = plt.figure(figsize=(16,5))
fig.add_subplot(2,2,1)
sns.kdeplot(x = datasetYeo['Wave_height'], y =  normData['Viscosity_cst'])
fig.add_subplot(2,2,2)
sns.kdeplot(x = datasetYeo['Engine-RPM'],y = normData['Viscosity_cst'])
fig.add_subplot(2,2,3)
sns.kdeplot(x = datasetYeo['SOG'], y = normData['Viscosity_cst'])
fig.add_subplot(2,2,4)
sns.kdeplot(x =datasetYeo['STW'],y = normData['Viscosity_cst'])
plt.tight_layout()
plt.show()

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

