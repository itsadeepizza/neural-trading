import pandas as pd
from model import LSTM_CELL
from merge_dataset import load_dataset
import torch
from tqdm import tqdm
import random



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

# Create the model
state_size = 20
lstm = LSTM_CELL(state_size=state_size, input_size=20)
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

# Split the dataset into training and testing
all_train = df_bit.iloc[:int(0.8*len(df_bit))]
test = df_bit.iloc[int(0.8*len(df_bit)):]

n_epochs = 100
train_size = 1_000
# Train the model
for epoch in range(n_epochs):
    random_idx = random.randint(0, len(all_train)-train_size)
    train = all_train.iloc[random_idx:random_idx+train_size]
    print(f'Epoch {epoch}')
    h = torch.ones(state_size)
    c = torch.ones(state_size)
    h = h.to(torch.device("cuda"))
    c = c.to(torch.device("cuda"))

    total_loss = 0
    for i in tqdm(range(len(train)-1)):
        x = train['price'].iloc[i]
        label = train['delta_price'].iloc[i]

        # Convert to tensor and send to GPU
        # Make x a tensor of shape (1)
        x = torch.ones(state_size) * x
        #label = torch.ones(state_size) * label
        label = torch.tensor([label]).float()
        x = x.to(torch.device("cuda"))
        label = label.to(torch.device("cuda"))

        # Forward pass
        h, c = lstm(h, c, x)
        y = h[0]

        # Compute the loss
        loss = torch.nn.MSELoss()(y, label)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the weights
        optimizer.step()

        # Reset the gradients for h and c
        h = h.detach()
        c = c.detach()
        if i % 300 == 0:
            print(f'current price {x[0]} - predicted price {y} - label {label} - loss {loss.item()}')

    print(f'Epoch {epoch} finished, Loss {total_loss/(len(train)-1)}')

    if epoch % 10 == 0:
        print(f'we will perform a test, WIP')



