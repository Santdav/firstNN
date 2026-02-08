import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable

df = pd.read_csv('student_habits_performance.csv')

EDA_MODULE: bool = True
menu_options: Dict[str, Callable] = {}


def EDA_module(dataframe):
    print("First 5 rows of the dataset:")
    print(dataframe.head())

    print("\nDataset Info:")
    print(dataframe.info())

    print("\nStatistical Summary:")
    print(dataframe.describe())
    
    print("\nMissing Values in Each Column:")
    print(dataframe.isnull().sum())

    print("\nDistribution of Numerical Features:")
    numerical_cols = dataframe.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        plt.figure()
        dataframe[col].hist(bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    print("\nCorrelation Matrix:")


def main_menu():
    print("\nMain Menu:")

if __name__ == "__main__":
    if EDA_MODULE:
        EDA_module(df)
    
    main_menu()