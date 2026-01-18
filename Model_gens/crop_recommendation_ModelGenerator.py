
import joblib
print("welcome to crop recommendation using LightGBM")

import pandas as pd
df=pd.read_csv("../datasets/Crop_recommendation.csv")


final=df[['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph',
       'rainfall', 'label']]


from sklearn.model_selection import train_test_split
X=df[['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph',
       'rainfall']]
Y=df[['label']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)



from lightgbm import LGBMClassifier

model=LGBMClassifier(n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42)
model.fit(x_train,y_train)
joblib.dump(model, "crop_recommendation_lgbm.pkl")

