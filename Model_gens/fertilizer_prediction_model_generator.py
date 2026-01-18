

print("Welcome to fertilizer Prediction")

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import joblib

df=pd.read_csv("../datasets/Fertilizer Prediction.csv")






final=df.rename(columns={"Temparature":"temperature","Humidity":"humidity","Moisture":"soil_moisture"})






X = final[['temperature','humidity','soil_moisture','Crop Type','Soil Type']]
y = final['Fertilizer Name']

cat_features = [X.columns.get_loc(col) for col in ['Crop Type', 'Soil Type']]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=y)

model = CatBoostClassifier(iterations=200,
    depth=8,
    learning_rate=0.08,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    cat_features=cat_features,
    random_seed=42,
    verbose=100)
model.fit(x_train, y_train)


joblib.dump(model, "fertilizer_recommendation_model.joblib")
print("Model Generated")
