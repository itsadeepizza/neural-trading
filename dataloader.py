from merge_dataset import load_dataset
import torch
# Load the dataset
df_bit = load_dataset()

# Convert price to float
df_bit['price'] = df_bit['price'].astype(float)



df_bit['next_price'] = df_bit['price'].shift(-1)
df_bit['delta_price'] = df_bit['next_price'] - df_bit['price']

# print mean and std
print(df_bit['price'].mean())
print(df_bit['price'].std())
# Normalize the price
df_bit['price'] = (df_bit['price'] - 65481) / 25580

# Normalize the delta price
print(df_bit['delta_price'].mean())
print(df_bit['delta_price'].std())
df_bit['delta_price'] = (df_bit['delta_price'] - 0) / 3.8

df_bit = df_bit[:-1]
print(df_bit.head())

