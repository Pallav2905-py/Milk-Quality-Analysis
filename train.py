import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import numpy as np

# Load dataset
df = pd.read_csv('milknew.csv')

# Print column names to debug
print("Original column names:", df.columns.tolist())

# Strip whitespace from column names
df.columns = df.columns.str.strip()
print("Cleaned column names:", df.columns.tolist())

# Drop specified columns: fat, turbidity, and taste
df = df.drop(['Fat', 'Turbidity', 'Taste'], axis=1)

# Separate features and target
X = df.drop('Grade', axis=1).values
y = df['Grade'].values

# Encode the target variable if it's not already numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(f"Target classes: {le.classes_} mapped to {np.unique(y)}")

# Check shapes
print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss')

# Define parameters for grid search
params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Grid search
print("\nTraining XGBoost model with grid search...")
grid_search = GridSearchCV(xgb_model, params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"XGBoost Accuracy: {accuracy:.4f}")
print(f"Best Parameters: {best_params}")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = best_model.feature_importances_
feature_names = ['pH', 'Temprature', 'Odor', 'Colour']
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Save the best model and label encoder
with open('milk_model.pkl', 'wb') as f:
    pickle.dump((best_model, le), f)
    
print(f"Model and label encoder saved as 'milk_model.pkl'")