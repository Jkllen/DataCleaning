import pandas as pd
import numpy as np

# Load Datasets
df1 = pd.read_csv("data/Road Accident Data.csv")
df2 = pd.read_csv("data/traffic_accidents.csv")
df3 = pd.read_csv("data/dataset_traffic_accident_prediction.csv")


# Standardize Column Names
def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


# Cleaning
def basic_cleaning(df):
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown")

    return df


# Normalize Category Value
def normalize_categories(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    return df

# Removing Outliers Using IQR
def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df

# Filter Based on Factors
factors = {
    "traffic_accident_data": ["accident_severity", "number_of_vehicles", "number_of_casualties"],
    "driver_condition_data": ["driver_age", "driver_gender", "alcohol_involved"],
    "weather_data": ["weather_conditions"],
    "lighting_condition": ["light_conditions"],
    "traffic_density_data": ["traffic_density"],
    "road_condition_data": ["road_surface_conditions"],
    "time_based_data": ["time", "date", "day_of_week"],
    "road_infrastructure_data": ["road_type", "junction_detail"],
    "vehicle_condition": ["vehicle_type", "vehicle_age"]
}

def filter_columns(df, factors):
    selected_cols = []
    for cols in factors.values():
        for col in cols:
            if col in df.columns:
                selected_cols.append(col)
    
    return df[selected_cols]

# Cleaning Process
def clean_process(df, name):
    print(f"\nProcessing {name}...")

    df = standardize_columns(df)
    df = basic_cleaning(df)
    df = normalize_categories(df)
    df = remove_outliers(df)
    df = filter_columns(df, factors)

    print(f"{name} shape after cleaning:", df.shape)
    return df

df1_clean = clean_process(df1, "Dataset 1")
df2_clean = clean_process(df2, "Dataset 2")
df3_clean = clean_process(df3, "Dataset 3")


# Saving Individual Dataset
df1_clean.to_csv("cleaned_dataset1.csv", index=False)
df2_clean.to_csv("cleaned_dataset2.csv", index=False)
df3_clean.to_csv("cleaned_dataset3.csv", index=False)

print("\n All datasets successfully cleaned.")