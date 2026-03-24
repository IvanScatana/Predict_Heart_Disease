import pandas as pd
import joblib
from model_utils import create_final_pipeline

import sklearn
sklearn.set_config(transform_output="pandas")

df = pd.read_csv('assets/heart.csv')
X, y = df.drop('HeartDisease', axis=1), df['HeartDisease']

final_pipeline = create_final_pipeline()
final_pipeline.fit(X, y)

joblib.dump(final_pipeline, 'final_pipeline.pkl')
print("✅ Модель сохранена!")