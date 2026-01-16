print("Welcome to Smart Farming Recommendation System")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

m = pd.read_csv('./soil_moisture.csv')
p = pd.read_csv('./Smart_Farming_Crop_Yield_2024.csv')

moist = m[['temperature','humidity','ph','rainfall','soil_moisture']]

phs = p[['humidity_%','temperature_C','rainfall_mm','pesticide_usage_ml','soil_moisture_%','soil_pH']].rename(
    columns={
        "humidity_%": "humidity",
        "temperature_C": "temperature",
        "soil_pH": "ph",
        "rainfall_mm": "rainfall",
        "soil_moisture_%": "soil_moisture"
    }
)

final = pd.concat([moist, phs], ignore_index=True)

final["irrigation_required"] = (
    (100 - final["soil_moisture"]) * 0.3 +
    final["temperature"] * 0.2 -
    final["rainfall"] * 0.05
)

final["irrigation_required"] = final["irrigation_required"].clip(lower=0)

OPTIMAL_PH = 6.5
final["ph_correction_amount"] = abs(final["ph"] - OPTIMAL_PH) * 10

X = final[['temperature', 'humidity', 'ph', 'rainfall', 'soil_moisture']]
y = final[['irrigation_required', 'ph_correction_amount']]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(xtrain, ytrain)

temp = float(input())
hum = float(input())
ph_val = float(input())
rain = float(input())
soil = float(input())

prediction = model.predict([[temp, hum, ph_val, rain, soil]])

irrigation = prediction[0][0]
ph_adjust = prediction[0][1]

print("Irrigation needed (mm):", round(irrigation, 2))
print("pH correction amount:", round(ph_adjust, 2))

joblib.dump(model, "smart_farming_recommendation.pkl")
