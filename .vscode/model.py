import pandas as pd
import seaborn as sns
import matplotlib as plt

file = 'voyage-data.csv'

voyage = pd.read_csv(file)


Xvar = voyage.drop(columns=['SOX', 'NOX'], axis=1)
yVar = (voyage[['SOX']])

print(Xvar, yVar)


sns.regplot(x=Xvar, y=yVar, data=voyage)
plt.show()




# print(voyage.info())
