import pandas as pd

df = pd.read_csv('amplified_UMIs.csv')

unique_orders = sorted(df["Order Number"].unique())
print(unique_orders, len(unique_orders))

