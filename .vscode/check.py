
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



print(vesseldf.range())