import pandas as pd 
import numpy as np 

df1 = pd.read_csv("cleaned_dataset1.csv") 
df2 = pd.read_csv("cleaned_dataset2.csv") 
df3 = pd.read_csv("cleaned_dataset3.csv") 
df4 = pd.read_csv("cleaned_dataset4.csv") 


final_cols = [ 
              "accident_severity", 
              "number_of_vehicles", 
              "number_of_casualties", 
              "crash_type", 
              "driver_alcohol", 
              "driver_age", 
              "driver_experience", 
              "weather_conditions", 
              "road_light_condition", 
              "traffic_density", 
              "road_conditions", 
              "time_of_day", 
              "road_defect", 
              "road_type", 
              "intersection_related", 
              "speed_limit", 
              "vehicle_type", 
              "maintenance_required", 
              "last_maintenance_required", 
              "failure_history", 
            ] 

def align_schema(df, cols): 
    for col in cols: 
        if col not in df.columns: df[col] = np.nan 
        return df[cols] 
    df1["source"] = "accident" 
    df2["source"] = "traffic" 
    df3["source"] = "prediction" 
    df4["source"] = "vehicle" 

df1 = align_schema(df1, final_cols) 
df2 = align_schema(df2, final_cols) 
df3 = align_schema(df3, final_cols) 
df4 = align_schema(df4, final_cols) 

df_all = pd.concat([df1, df2, df3, df4], ignore_index=True) 

df_all = df_all.fillna("unknown") 

df_all.to_csv("Stackingfuzzy_dataset.csv", index=False) 

print("Unified dataset created:", df_all.shape)