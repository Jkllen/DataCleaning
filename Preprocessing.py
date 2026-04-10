import pandas as pd
import numpy as np

# LOAD DATASETS
df1 = pd.read_csv("data/Road Accident Data.csv")
df2 = pd.read_csv("data/traffic_accidents.csv")
df3 = pd.read_csv("data/dataset_traffic_accident_prediction.csv")
df4 = pd.read_csv("data/revised_logistics_dataset_V2.csv")


# STANDARDIZATION
def standardize_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


# BASIC CLEANING
def basic_cleaning(df):
    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown")

    return df


# NORMALIZATION
def normalize_categories(df):
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):

            df[col] = df[col].astype(str)
            df[col] = df[col].str.strip().str.lower()

    return df


# OUTLIER REMOVAL (Only for Numerical Variables)
def remove_outliers(df, cols):
    for col in cols:
        if col in df.columns and df[col].nunique() > 10:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


# FINAL FUZZY VARIABLES
factors = {
    "traffic_accident_data": [
        "accident_severity",
        "number_of_vehicles",
        "number_of_casualties",
        "crash_type"
    ],

    "driver_condition_data": [
        "driver_alcohol",
        "driver_age",
        "driver_experience"
    ],

    "weather_data": [
        "weather_conditions"
    ],

    "lighting_condition": [
        "road_light_condition"
    ],

    "traffic_density_data": [
        "traffic_density"
    ],

    "road_condition_data": [
        "road_conditions"
    ],

    "time_based_data": [
        "time_of_day"
    ],

    "road_infrastructure_data": [
        "road_defect",
        "road_type",
        "intersection_related",
        "speed_limit"
    ],

    "vehicle_condition": [
        "vehicle_type",
        "vehicle_age",
        "maintenance_required",
        "last_maintenance_required",
        "failure_history",
        "brake_condition"
    ]
}


# COLUMN MAPPINGS

column_mapping_df1 = {
    "accident_severity": "accident_severity",
    "number_of_vehicles": "number_of_vehicles",
    "number_of_casualties": "number_of_casualties",
    "light_conditions": "road_light_condition",
    "road_surface_conditions": "road_conditions",
    "time": "time_of_day",
    "vehicle_type": "vehicle_type",
    "speed_limit": "speed_limit",
    "junction_detail": "intersection_related"
}


column_mapping_df2 = {
    "most_severe_injury": "accident_severity",
    "num_units": "number_of_vehicles",
    "injuries_total": "number_of_casualties",
    "weather_condition": "weather_conditions",
    "lighting_condition": "road_light_condition",
    "roadway_surface_cond": "road_conditions",
    "crash_hour": "time_of_day",
    "crash_date": "date",
    "crash_type": "crash_type",
    "intersection_related_i": "intersection_related",
    "traffic_control_device": "road_defect"
}


column_mapping_df3 = {
    "weather": "weather_conditions",
    "road_condition": "road_conditions",
    "road_light_condition": "road_light_condition",
    "traffic_density": "traffic_density",
    "time_of_day": "time_of_day",
    "vehicle_type": "vehicle_type",
    "accident_severity": "accident_severity",
    "driver_alcohol": "driver_alcohol",
    "driver_age": "driver_age",
    "driver_experience": "driver_experience",
    "road_type": "road_type",
    "speed_limit": "speed_limit"
}


column_mapping_df4 = {
    "vehicle_type": "vehicle_type",
    "year_of_manufacture": "vehicle_age",
    "last_maintenance_date": "last_maintenance_required",
    "maintenance_required": "maintenance_required",
    "failure_history": "failure_history",
    "brake_condition": "brake_condition"
}


# FILTER FUNCTION
def filter_columns(df, factors):
    keep_cols = []

    for cols in factors.values():
        for col in cols:
            if col in df.columns:
                keep_cols.append(col)

    return df.loc[:, list(dict.fromkeys(keep_cols))]


# MAIN
def clean_process(df, name, mapping=None, outlier_cols=None):

    print(f"\nProcessing {name}...")

    df = standardize_columns(df)

    if mapping:
        df = df.rename(columns=mapping)

    df = basic_cleaning(df)
    df = normalize_categories(df)

    # If null, derive vehicle age
    if "vehicle_age" not in df.columns and "year_of_manufacture" in df.columns:
        df["vehicle_age"] = 2026 - df["year_of_manufacture"]

    # Safe outlier removal only on selected numeric fields
    if outlier_cols:
        df = remove_outliers(df, outlier_cols)

    df = filter_columns(df, factors)

    print(f"{name} shape:", df.shape)
    return df


# RUN CLEANING
df1_clean = clean_process(df1, "Dataset 1", column_mapping_df1, outlier_cols=["number_of_vehicles", "number_of_casualties"])
df2_clean = clean_process(df2, "Dataset 2", column_mapping_df2, outlier_cols=["number_of_vehicles", "number_of_casualties"])
df3_clean = clean_process(df3, "Dataset 3", column_mapping_df3, outlier_cols=None)
df4_clean = clean_process(df4, "Dataset 4", column_mapping_df4, outlier_cols=["vehicle_age"])


# SAVE OUTPUT
df1_clean.to_csv("cleaned_dataset1.csv", index=False)
df2_clean.to_csv("cleaned_dataset2.csv", index=False)
df3_clean.to_csv("cleaned_dataset3.csv", index=False)
df4_clean.to_csv("cleaned_dataset4.csv", index=False)

print("\nAll datasets cleaned.")