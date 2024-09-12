from merge_dataset import load_dataset
import torch
# Load the dataset
df_bit = load_dataset()

# Convert price to float
df_bit['price'] = df_bit['price'].astype(float)



df_bit['next_price'] = df_bit['price'].shift(-1)



