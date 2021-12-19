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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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

print(f"{vesseldf.info()}\n {envdf.info()}\n{fueldf.info()}")


# Xvar = voyage.drop(columns=['SOX', 'NOX'], axis=1)
# yVar = (voyage[['SOX', 'NOX']])




# sns.regplot(x=Xvar['Engine-RPM'], y=yVar['SOX'], data=voyage)

# Xvar.loc[:, 'DFT'] = 12.1 #from AIS data

# plt.show()
# print(Xvar, yVar)

# print(voyage.info())
