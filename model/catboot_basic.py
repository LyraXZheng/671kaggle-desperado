import os
import numpy as np
import pandas as pd
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from catboost import CatBoostClassifier
from joblib import dump

# Provided parameters
best_params = {
    'iterations': 2000,
    'depth': 9,
    'learning_rate': 0.015694383222863578,
    'l2_leaf_reg': 7.832872486793348,
    'border_count': 141,
    'bagging_temperature': 0.30343401538183157,
    'random_strength': 0.05737372288204851,
    'random_state': 42,
    'early_stopping_rounds': 50,
    'task_type': 'GPU',
    'devices': '0'
}

# Load data
df_train = pd.read_csv('../../data/important data/df_train_fs_together.csv')
df_test = pd.read_csv('../../data/important data/df_test_fs_together.csv')
df_test.set_index('id', inplace=True)

X = df_train.drop(columns=['price'], axis=1)
y = df_train['price']

# Split the data into training and testing sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the CatBoost model
print("Training CatBoost with provided parameters...")
model = CatBoostClassifier(**best_params)
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# Save the model
model_path = "best_catboost_model.joblib"
dump(model, model_path)
print(f"CatBoost model saved to: {os.path.abspath(model_path)}")

# Evaluate the model on the test set
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions on the test dataset
test_predictions = model.predict(df_test).flatten()

# Save predictions to a CSV file
output_path = '../../data/result data/test_predict_catboost.csv'
pd.DataFrame({'id': df_test.index, 'price': test_predictions}).to_csv(output_path, index=False)
print(f"Test predictions saved to: {os.path.abspath(output_path)}")
