"when dataset is genrated from python script, a lot of csv files are generated. We need to merge them into one file"


import pandas as pd
import os

path = 'dataset'

def load_dataset():

    #  Load all the csv files ending with .csv in the dataset folder
    
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    # Read all the csv files and store them in a list
    dfs = [pd.read_csv(os.path.join(path, f), sep=';') for f in files]
    
    # Concatenate all the dataframes in the list
    df = pd.concat(dfs)
    
    # Order the dataframe by timestamp
    df = df.sort_values(by='timestamp')
    
    # keep only bitcoin
    df_bit = df[df['coin'] == 'bitcoin']

    # Convert price to float
    df_bit['price'] = df_bit['price'].astype(float)
    
    return df_bit

if __name__ == "__main__":
    df_bit = load_dataset()
    print(df_bit.head())
    df_bit = load_dataset()
    # Plot a graph
    # Reduce sampling to 30 minutes
    df_bit = df_bit[::1800]
    
    
    import matplotlib.pyplot as plt
    
    # The timestamp is formatted as 2024-04-09 23:22:18:542047
    
    df_bit['datetime'] = pd.to_datetime(df_bit['timestamp'], format='%Y-%m-%d %H:%M:%S:%f')
    
    plt.plot(df_bit['datetime'], df_bit['price'])
    # Add x-ticks
    plt.xticks(rotation=30)
    
    # autoformat the plot
    plt.tight_layout()
    
    
    plt.xlabel('Timestamp')
    
    plt.show()
