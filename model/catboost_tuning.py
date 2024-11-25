import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from optuna import create_study
from tqdm import tqdm
from joblib import dump
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count

# Check GPU availability
gpu_count = get_gpu_device_count()
print(f"Number of GPUs available for CatBoost: {gpu_count}")

# Global parameters
OPTUNA_TRIALS = 100  # Number of Optuna trials

# Load and preprocess data
df_train = pd.read_csv('../../data/important data/train_final1.csv')
df_test = pd.read_csv('../../data/important data/train_final2.csv')
# df_test.set_index('id', inplace=True)

X = df_train.drop(columns=['price'], axis=1)
y = df_train['price']

# Split the data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Dictionary to store model performance
model_rmse = {}

# CatBoost Classifier optimization with Optuna
print("Optimizing CatBoost with Optuna...")
best_rmse = float('inf')
best_accuracy = 0


def objective_cb(trial):
    """Objective function for Optuna to optimize CatBoost hyperparameters."""
    global best_rmse, best_accuracy

    # Define hyperparameter search space
    params = {
        'iterations': 2000,
        'depth': trial.suggest_int('depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.01, 5.0),
        'random_strength': trial.suggest_float('random_strength', 0.01, 1.0),
        'verbose': 0,
        'random_state': 42,
        'early_stopping_rounds': 50,
        'task_type': 'GPU',
        'devices': '0',
        'gpu_ram_part': 0.9,
        'max_ctr_complexity': 1,
    }

    # Train the model
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)

    # Update best scores
    if rmse < best_rmse:
        best_rmse = rmse
        best_accuracy = accuracy

    print(f"Trial RMSE: {rmse:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Best RMSE: {best_rmse:.4f}, Best Accuracy: {best_accuracy:.4f}")
    return rmse


def callback_cb(study, trial):
    """Callback to update the progress bar."""
    pbar.update(1)


with tqdm(total=OPTUNA_TRIALS, desc="CatBoost Optimization Progress") as pbar:
    study_cb = create_study(direction="minimize")
    study_cb.optimize(objective_cb, n_trials=OPTUNA_TRIALS, callbacks=[callback_cb])

# Best parameters
best_params_cb = study_cb.best_params
print(f"Best Parameters: {best_params_cb}")

# Train final model with best parameters
final_model = CatBoostClassifier(
    **best_params_cb, verbose=0, random_state=42, task_type='GPU', devices='0'
)
final_model.fit(X_train, y_train)

# Save the trained model
model_path = "best_catboost_model.joblib"
dump(final_model, model_path)
print(f"CatBoost model saved to: {os.path.abspath(model_path)}")

# Evaluate final model
final_predictions = final_model.predict(X_test)
model_rmse['CatBoost'] = np.sqrt(mean_squared_error(y_test, final_predictions))

# Print model performance
print("\nModel Performance:")
for model, rmse in model_rmse.items():
    print(f"{model} - RMSE: {rmse:.4f}")

# Make predictions on test data
test_predictions = final_model.predict(df_test).flatten()
output_path = 'result/test_predict_catboost.csv'
pd.DataFrame({'id': df_test.index, 'price': test_predictions}).to_csv(output_path, index=False)
print(f"Test predictions saved to: {os.path.abspath(output_path)}")
