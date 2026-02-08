import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable



EDA_MODULE: bool = False
CLEANING_MODULE: bool = False


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
    df = pd.read_csv('student_habits_performance.csv')

    if CLEANING_MODULE:
        #drop id column
        df.drop('student_id', axis=1, inplace=True)

        #impute mode for parental_education_level
        df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])

        #binary encode
        binary_mapping = {'Yes': 1, 'No': 0}
        df['extracurricular_participation'] = df['extracurricular_participation'].map(binary_mapping)
        df['part_time_job'] = df['part_time_job'].map(binary_mapping)

        # all other cleaning are model dependant and will be done in the modeling module

        #save cleaned data for modeling module
        df.to_csv('cleaned_data.csv', index=False)

    if EDA_MODULE:
        EDA_module(df)
    
    main_menu()
