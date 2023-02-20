import pandas as pd
from pathlib import Path
import random

def make_data_set(dataset, n_assets):

    # Set variables based on dataset selected.
    if dataset == "bonds":
        file_name = "HistoricalData.csv"
        col_start = 2
        max_assets = 92
    elif dataset == "prices":
        file_name = "prices_df.csv"
        col_start = 1
        max_assets = 4
    else:
        print("Unknown dataset. Please enter either 'bonds', or 'prices'.")

    print("Data source file: ", file_name)
    
    # Open file and read data.
    path = Path(__file__).parent.parent / "data"
    file = path / file_name
    data = pd.read_csv(file)

    # Randomly sample assets.
    sample = random.sample(range(1, max_assets+1), min(n_assets, max_assets))
    returns = data.iloc[:, sample]
          
    # Fill missing values with last previous observation.
    returns.fillna(method='bfill')

    return returns