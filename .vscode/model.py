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

fueldf["Date"] = pd.to_datetime(fueldf["Date"])
fueldf["Time"] = pd.to_datetime(fueldf["Time"])

# print(f"{vesseldf.info()}\n {envdf.info()}\n{fueldf.info()}")

CombinedMetrics = pd.DataFrame()
CombinedMetrics = pd.concat([vesseldf, envdf, fueldf], axis=1) #joined all datasources into one DF for easier manipulation

# Setting Output Variables 
yVar = pd.DataFrame()
yVar = (CombinedMetrics[['SOX', 'NOX', 'Viscosity_cst']])

# Setting Input Variables 
Xvar = CombinedMetrics.drop(columns=['SOX', 'NOX', 'Date', 'Time', 'Viscosity_cst'], axis=1)



# define min max scaler
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(Xvar)
print(scaled)


# fig = plt.figure(figsize=(10,7))
# fig.add_subplot(2,1,1)
# sns.distplot(yVar['Viscosity_cst'])
# fig.add_subplot(2,1,2)
# sns.boxplot(yVar['Viscosity_cst'])
# plt.show()
# plt.tight_layout()



# X = Xvar.iloc[:,0:16]  #independent columns
# y = yVar.iloc[:,-1]    #viscosity
# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
# model = ExtraTreesClassifier()
# model.fit(X,y)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(6).plot(kind='barh')
# plt.title('Feature Importance')
# plt.tight_layout()
# plt.show()




# plt.figure(figsize=(12,10))
# cor = CombinedMetrics.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.tight_layout()
# plt.show()

# fig = plt.figure(figsize=(16,5))
# fig.add_subplot(2,2,1)
# sns.scatterplot(Xvar['Wave_height'], yVar['Viscosity_cst'])
# fig.add_subplot(2,2,2)
# sns.scatterplot(Xvar['F.O_Temp_Celcius'],yVar['Viscosity_cst'])
# fig.add_subplot(2,2,3)
# sns.scatterplot(Xvar['L.O.Main_Temp_Celcius'],yVar['Viscosity_cst'])
# fig.add_subplot(2,2,4)
# sns.scatterplot(Xvar['L.O.Main_Pressure_kgm3'],yVar['Viscosity_cst'])
# plt.tight_layout()
# plt.show()

# fig = plt.figure(figsize=(15,7))
# fig.add_subplot(2,2,1)
# sns.countplot(Xvar['Wave_height'])
# fig.add_subplot(2,2,2)
# sns.countplot(Xvar['F.O_Temp_Celcius'])
# fig.add_subplot(2,2,3)
# sns.countplot((Xvar['L.O.Main_Temp_Celcius']))
# fig.add_subplot(2,2,4)
# sns.countplot(Xvar['L.O.Main_Pressure_kgm3'])
# plt.tight_layout()
# plt.show()



# ###############################################################################
# F    E    A    T    U    R    E   •••••  S    E    L   L   E   C   T   I   O   N

# sns.distplot(Xvar)

# plt.tight_layout()
# plt.show()

# sns.regplot(x=Xvar['Engine-RPM'], y=yVar['SOX'], data=voyage)

# Xvar.loc[:, 'DFT'] = 12.1 #from AIS data


# print(Xvar, yVar)
