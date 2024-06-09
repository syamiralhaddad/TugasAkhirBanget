import pandas as pd
import warnings
import joblib
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)

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

# Handle categorical columns (traffic_type, variation_traffic_from, protocol)
categorical_features = ['variation_traffic_from', 'protocol']
numeric_features = [feature for feature in features if feature not in categorical_features]

# Preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Define hyperparameter grids for each model
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf']
}

param_grid_knn = {
    'n_neighbors': [5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Initialize base models with hyperparameter tuning
svm_model = SVC(probability=True)  # probability=True to enable predict_proba for stacking
knn_model = KNeighborsClassifier()

# Initialize meta-model
meta_model = LogisticRegression()

# Define K-Fold cross-validation with fewer folds
kfold = KFold(n_splits=2, shuffle=True, random_state=42)

# Initialize empty arrays to store meta-features
meta_features_train = []

# Define scoring metrics
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_scores = {metric: [] for metric in scoring}

# Iterate through base models
for model, param_grid in [(svm_model, param_grid_svm), (knn_model, param_grid_knn)]:
    # Create pipeline with preprocessor and model
    pipeline = make_pipeline(preprocessor, GridSearchCV(model, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1))

    # Perform cross-validation predictions
    cv_pred = cross_val_predict(pipeline, X, y, cv=kfold, method='predict_proba', n_jobs=-1)

    # Append predictions as meta-features
    meta_features_train.append(pd.DataFrame(cv_pred))

# Concatenate meta-features across base models
meta_features_train = pd.concat(meta_features_train, axis=1)

# Fit the meta-model (logistic regression) with the meta-features
meta_model.fit(meta_features_train, y)

# Inverse transform predicted labels to original class labels using the stacked ensemble model
meta_pred = meta_model.predict(meta_features_train)
predicted_traffic_type = label_encoder.inverse_transform(meta_pred)

# Assign predicted values to the entire DataFrame
data['Predicted Traffic Type'] = predicted_traffic_type
predicted_traffic_type_df = pd.DataFrame(data.iloc[1:1018])

# Define K-Fold cross-validation with fewer folds
kfold = KFold(n_splits=2, shuffle=True, random_state=42)

# Calculate cross-validation scores
meta_cv_scores = cross_val_score(meta_model, meta_features_train, y, cv=kfold, scoring='accuracy', n_jobs=-1)

# Perform cross-validation with meta-features and meta-model
meta_cv_pred = cross_val_predict(meta_model, meta_features_train, y, cv=kfold, method='predict', n_jobs=-1)

# Calculate evaluation metrics
accuracy = accuracy_score(y, meta_cv_pred)
precision = precision_score(y, meta_cv_pred, average='weighted')
recall = recall_score(y, meta_cv_pred, average='weighted')
f1 = f1_score(y, meta_cv_pred, average='weighted')
meta_cv_mean_accuracy = np.mean(meta_cv_scores)

# Create a DataFrame with the metrics data
metrics_data = {
"Metric": ["Accuracy", "Precision", "Recall", "F1-score", "Cross-Val"],
"Value": [accuracy, precision, recall, f1, meta_cv_mean_accuracy]
}

metrics_df = pd.DataFrame(metrics_data)

#joblib.dump(meta_model, 'st1_model.joblib')
#oblib.dump(meta_features_train, 'st1_features.joblib')
joblib.dump(metrics_df, 'st1_metrics.joblib')
joblib.dump(predicted_traffic_type_df, 'st1_traffic_type.joblib')