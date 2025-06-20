import os
import sys


DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(DIR)
print(DIR)
os.chdir(DIR)
sys.path.append(DIR)

import numpy as np
import pandas as pd

import src.data.Read_data as read_data

def calculate_and_print_stats(dataset_name, all_series):
    """
    Calculates and prints statistics for a given list of time series.
    
    Args:
        dataset_name (str): The name of the dataset for printing.
        all_series (list): A list of numpy arrays, where each array is a time series.
    """


    # Calculate the length of each series
    lengths = [len(ts) for ts in all_series]
    
    # Calculate statistics
    stats = {
        "No. of series": len(all_series),
        "Min. length": min(lengths),
        "Max. length": max(lengths),
        "Mean length": np.mean(lengths),
        "Std. dev. length": np.std(lengths),
    }

    # Print the formatted table
    print("-" * 40)
    print(f"Statistics for: {dataset_name}")
    print("-" * 40)
    print(f"{'Statistic':<20} | {'Value':>15}")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, int):
            print(f"{key:<20} | {value:>15}")
        else:
            print(f"{key:<20} | {value:>15.1f}")
    print("-" * 40)
    print("\n")
    output_dir = "tables"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a pandas DataFrame from the stats dictionary
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    stats_df.index.name = 'Statistic'
    
    # Generate filename and save the DataFrame
    csv_filename = dataset_name.replace(" ", "_").lower() + "_stats.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    stats_df.to_csv(csv_filepath)
    print(f"Statistics for {dataset_name} saved to '{csv_filepath}'")


_, _, testset_m3 = read_data.read_data("M3_Monthly")
calculate_and_print_stats("M3 Monthly", testset_m3)

_, _, testset_m4 = read_data.read_data("M4_Monthly")
calculate_and_print_stats("M4 Monthly", testset_m4)
