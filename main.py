import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable

from Models.LinearRegression import LinearRegression


EDA_MODULE: bool = False
CLEANING_MODULE: bool = True




def linear_regression_module():
    print("Linear Regression model will be implemented here.")

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

    while True:
        print("\n===Main Menu===")
        for index, option in enumerate(menu_options.keys(), start=1):
            print(f"{index}. {option}")
        choice = input("Select an option (or 'q' to quit): \n>").lower().strip()
        try:
            if choice.lower() == 'q':
                print("Exiting the program.")
                return;
            if int(choice) in range(1, len(menu_options) + 1):
                selected_option = list(menu_options.keys())[int(choice) - 1]
                menu_options[selected_option]()

        except ValueError as e:
            print("Not a valid option, try again")


menu_options: Dict[str, Callable] = {
    "Linrear Regression": linear_regression_module,
}
        
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

        # encoding for gender
        df = pd.get_dummies(df, columns=['gender'], prefix='gender', dtype=int)

        # Encoding for categorical ordinal features
        diet_quality_mapping = {'Poor': 1, 'Fair': 2, 'Good': 3}
        df['diet_quality'] = df['diet_quality'].map(diet_quality_mapping)

        parental_education_level_mapping = {
            "None": 0,
            'High School': 1,
            'Bachelor': 2,
            'Master': 3,
        }
        df['parental_education_level'] = df['parental_education_level'].map(parental_education_level_mapping)

        internet_quality_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
        df['internet_quality'] = df['internet_quality'].map(internet_quality_mapping)

        #save cleaned data for modeling module
        df.to_csv('cleaned_data.csv', index=False)

        print("Data cleaning completed and saved to 'cleaned_data.csv'.")

    if EDA_MODULE:
        EDA_module(df)
    
    df = pd.read_csv('cleaned_data.csv')

    #main_menu()
