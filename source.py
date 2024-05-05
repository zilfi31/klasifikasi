import pandas as pd

df = pd.read_csv("bank.csv")

age = df['age'].unique()
job = df['job'].unique()
marital = df['marital'].unique()
education = df['education'].unique()
default = df['default'].unique()
balance = df['balance'].unique()
housing = df['housing'].unique()
loan = df['loan'].unique()
y = df['y'].unique()
