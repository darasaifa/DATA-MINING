import pandas as pd

df = pd.read_csv("MarketingTarget.csv", sep=";")


print(df.info())

print(df.head())
