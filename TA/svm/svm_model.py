import pandas as pd
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore", category=Warning)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv('network_traffic_5G_final_data.csv')

# Select features and target variable
features = ['variation_traffic_from', 'protocol', 'packet_loss', 'throughput', 'latency', 'jitter', 'ping', 'packet_length']
target = 'traffic_type'

# Encode categorical target variable 'traffic_type' using LabelEncoder
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])
X = data[features]
y = data[target]

# Separate numerical and categorical features
categorical_features = ['variation_traffic_from', 'protocol']
numerical_features = [feature for feature in features if feature not in categorical_features]

# Preprocessing pipeline for numerical features
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
# Preprocessing pipeline for categorical features
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

# Combine the preprocessing steps for both numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the data
scaled_features = preprocessing_pipeline.fit_transform(X)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# Initialize SVM classifier
svm_model = SVC()

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(scaled_features, y)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Initialize SVM classifier with the best hyperparameters
best_svm_model = SVC(**best_params)

# Fit the model
best_svm_model.fit(scaled_features, y)

# Predictions using the voting classifier
y_pred = best_svm_model.predict(scaled_features)

# Inverse transform predicted labels to original class labels
predicted_traffic_type = label_encoder.inverse_transform(y_pred)
data['Predicted Traffic Type'] = predicted_traffic_type
predicted_traffic_type_df = pd.DataFrame(data.iloc[1:1018])

# Perform cross-validation on the SVM model
cv_scores = cross_val_score(best_svm_model, scaled_features, y, cv=5, scoring='accuracy')

# Calculate mean cross-validation accuracy
mean_cv_accuracy = np.mean(cv_scores)

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

# Create a DataFrame with the metrics data
metrics_data = {
"Metric": ["Accuracy", "Precision", "Recall", "F1-score", "Cross-Val"],
"Value": [accuracy, precision, recall, f1, mean_cv_accuracy]
}

metrics_df = pd.DataFrame(metrics_data)

#joblib.dump(best_svm_model, 'svm_model.joblib')
joblib.dump(metrics_df, 'svm_metrics.joblib')
joblib.dump(predicted_traffic_type_df, 'svm_traffic_type.joblib')