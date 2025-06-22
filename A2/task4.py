import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('king_rook_vs_king.csv')

# this is to transfor the labels into a numerical format
def map_target(value):
    if value == 'draw':
        return 0
    elif value in ['zero', 'one', 'two', 'three', 'four']:
        return 1
    elif value in ['five', 'six', 'seven', 'eight']:
        return 2
    elif value in ['nine', 'ten', 'eleven', 'twelve']:
        return 3
    elif value in ['thirteen', 'fourteen', 'fifteen', 'sixteen']:
        return 4
    return -1 # Should not happen
df['target'] = df['white_depth_of_win'].apply(map_target)


# Feature Engineering: Ordinal Encoding for file columns
file_cols = ['white_king_file', 'white_rook_file', 'black_king_file']
file_order = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ordinal_encoder = OrdinalEncoder(categories=[file_order] * len(file_cols))
df[file_cols] = ordinal_encoder.fit_transform(df[file_cols]) + 1 # Add 1 to map to 1-8 instead of 0-7

# Define features (X) and target (y)
features = [
    'white_king_file', 'white_king_rank',
    'white_rook_file', 'white_rook_rank',
    'black_king_file', 'black_king_rank'
]
X = df[features]
y = df['target']



# Run Optuna for Hyperparameter Optimization

print("\nSetting up Optuna for hyperparameter tuning...")

# Load Optuna configuration from YAML file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

N_TRIALS = config['n_trials']

def objective(trial):
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', **config['params']['n_estimators']),
        'learning_rate': trial.suggest_float('learning_rate', **config['params']['learning_rate']),
        'num_leaves': trial.suggest_int('num_leaves', **config['params']['num_leaves']),
        'max_depth': trial.suggest_int('max_depth', **config['params']['max_depth']),
        'reg_alpha': trial.suggest_float('reg_alpha', **config['params']['reg_alpha']),
        'reg_lambda': trial.suggest_float('reg_lambda', **config['params']['reg_lambda']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', **config['params']['colsample_bytree']),
        'subsample': trial.suggest_float('subsample', **config['params']['subsample']),
        'class_weight': 'balanced', 
        'n_jobs': -1,
        'random_state': 42
    }

    # Use Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

        preds = model.predict(X_val)
        # We optimize for balanced accuracy
        balanced_acc = balanced_accuracy_score(y_val, preds)
        scores.append(balanced_acc)

    return np.mean(scores)

# Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
print(f"Running Optuna optimization for {N_TRIALS} trials...")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Optimization finished.")
print("Best trial:")
best_trial = study.best_trial
print(f"  Value (Balanced Accuracy): {best_trial.value:.4f}")
print("  Best hyperparameters: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")


# Train Final Model and Evaluate on Test Set

print("\nTraining final model with best hyperparameters...")

best_params = best_trial.params
best_params['objective'] = 'multiclass'
best_params['metric'] = 'multi_logloss'
best_params['class_weight'] = 'balanced'
best_params['random_state'] = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)


# test the final model on the test set
print("\n--- Final Model Evaluation on Hold-out Test Set ---")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

final_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Final Balanced Accuracy: {final_balanced_accuracy:.4f}\n")

