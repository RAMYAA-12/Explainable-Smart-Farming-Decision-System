import pandas as pd

df = pd.read_csv("data/crop.csv")

df_irrigation = df[["temperature", "humidity", "rainfall"]].copy()

# Simple realistic formula
df_irrigation["water_needed"] = 100 - df_irrigation["rainfall"] * 0.3

df_irrigation.to_csv("data/irrigation.csv", index=False)

print("Irrigation dataset created!")