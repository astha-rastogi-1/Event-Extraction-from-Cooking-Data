import pandas as pd

# Check for duplicates within the individual csv files

df = pd.read_csv("bake.csv")

df = df.drop(columns=['Unnamed: 0'])
df.drop_duplicates(subset='URL', keep='first', inplace=True)

df.to_csv("new_bake.csv") 