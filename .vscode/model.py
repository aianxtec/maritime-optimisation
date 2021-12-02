import pandas as pd

file = 'voyage-data.csv'

voyage = pd.read_csv(file)

df = pd.DataFrame()

print(voyage)