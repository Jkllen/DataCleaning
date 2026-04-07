import pandas as pd
import numpy as np

# Load Datasets
df1 = pd.read_csv("data/Road Accident Data.csv")
df2 = pd.read_csv("data/traffic_accidents.csv")
df3 = pd.read_csv("data/dataset_traffic_accident_prediction.csv")
df4 = pd.read_csv("data/revised_logistics_dataset_V2.csv")

# For d4 only
df4.columns = df4.columns.str.strip().str.lower().str.replace(" ", "_")

# Create vehicle_age
df4['vehicle_age'] = 2026 - df4['year_of_manufacture']

# Convert date column properly
df4['last_maintenance_date'] = pd.to_datetime(df4['last_maintenance_date'], errors='coerce')

# Standardize Column Names
def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# Basic Cleaning: remove duplicates and fill missing values
def basic_cleaning(df):
    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(pd.Timestamp("1970-01-01"))
        
        else:
            df[col] = df[col].fillna("unknown")

    return df

# Normalize Category Values
def normalize_categories(df):
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].str.strip().str.lower()
    return df

# Remove Outliers Using IQR
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

# Defined Factors
factors = {
    "traffic_accident_data": ["accident_severity", "number_of_vehicles", "number_of_casualties"],
    "driver_condition_data": ["driver_age", "driver_gender", "alcohol_involved"],
    "weather_data": ["weather_conditions"],
    "lighting_condition": ["light_conditions"],
    "traffic_density_data": ["traffic_density"],
    "road_condition_data": ["road_surface_conditions"],
    "time_based_data": ["time", "date", "day_of_week"],
    "road_infrastructure_data": ["road_type", "junction_detail"],
    "vehicle_condition": ["vehicle_type", "vehicle_age", "engine_temperature", "tire_pressure", "brake_condition", "oil_quality", "battery_status", "vibration_levels", "last_maintenance_date", "maintenance_required", "failure_history"]
}

# Column mapping for Dataset 2 Matching the Variables of the Main Factors
column_mapping_df2 = {
    "most_severe_injury": "accident_severity",
    "num_units": "number_of_vehicles",
    "injuries_total": "number_of_casualties",
    "weather_condition": "weather_conditions",
    "lighting_condition": "light_conditions",
    "roadway_surface_cond": "road_surface_conditions",
    "crash_date": "date",
    "crash_day_of_week": "day_of_week",
    "crash_hour": "time"
}


# Filter only expert-selected columns
def filter_columns(df, factors):
    selected_cols = []
    for cols in factors.values():
        for col in cols:
            if col in df.columns:
                selected_cols.append(col)
    return df.loc[:, list(dict.fromkeys(selected_cols))]

# Full Cleaning Process
def clean_process(df, name, column_mapping=None, remove_outlier_flag=True):
    print(f"\nProcessing {name}...")

    # 1. Standardize all columns
    df = standardize_columns(df)
    
    # 2. Column Mapping
    if column_mapping:
        df = df.rename(columns=column_mapping)
    # 3. Basic cleaning
    df = basic_cleaning(df)
    # 4. Normalize categorical values
    df = normalize_categories(df)
    # 5. Remove outliers
    if remove_outlier_flag:
        df = remove_outliers(df)
    # 6. Expert-selected columns (based dun sa dataset natin na pinakita sa mga na-interviewed natin)
    df = filter_columns(df, factors)

    print(f"{name} shape after cleaning:", df.shape)
    return df

# Clean each dataset
df1_clean = clean_process(df1, "Dataset 1")
df2_clean = clean_process(df2, "Dataset 2", column_mapping_df2)
df3_clean = clean_process(df3, "Dataset 3")
df4_clean = clean_process(df4, "Dataset 4", remove_outlier_flag=False)

# Save cleaned datasets
df1_clean.to_csv("cleaned_dataset1.csv", index=False)
df2_clean.to_csv("cleaned_dataset2.csv", index=False)
df3_clean.to_csv("cleaned_dataset3.csv", index=False)
df4_clean.to_csv("cleaned_dataset4.csv", index =False)

print("\nAll datasets successfully cleaned.") 