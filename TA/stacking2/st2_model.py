import pandas as pd
import joblib
import warnings
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

# Define simplified hyperparameter grids for each model
param_grid_rf = {
    'classifier__n_estimators': [10],
    'classifier__max_depth': [5]
}

param_grid_svm = {
    'classifier__C': [1],
    'classifier__gamma': ['scale']
}

param_grid_knn = {
    'classifier__n_neighbors': [3],
    'classifier__weights': ['uniform']
}

# Initialize base models
rf_model = RandomForestClassifier()
svm_model = SVC(probability=True)
knn_model = KNeighborsClassifier()

# Create pipelines with hyperparameter tuning
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', svm_model)
])

knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', knn_model)
])

# Define K-Fold cross-validation
kfold = KFold(n_splits=2, shuffle=True, random_state=42)

# Initialize empty arrays to store meta-features
meta_features_train = []

# Iterate through base models
for pipeline, param_grid in zip([rf_pipeline, svm_pipeline, knn_pipeline], [param_grid_rf, param_grid_svm, param_grid_knn]):
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
    cv_pred = cross_val_predict(grid_search, X, y, cv=kfold, method='predict_proba', n_jobs=-1)
    meta_features_train.append(pd.DataFrame(cv_pred))

# Concatenate meta-features across base models
meta_features_train = pd.concat(meta_features_train, axis=1)

# Fit logistic regression on meta-features
meta_model = LogisticRegression()
meta_model.fit(meta_features_train, y)

# Inverse transform predicted labels to original class labels using the stacked ensemble model
meta_pred = meta_model.predict(meta_features_train)
predicted_traffic_type = label_encoder.inverse_transform(meta_pred)

# Assign predicted values to the entire DataFrame
data['Predicted Traffic Type'] = predicted_traffic_type
predicted_traffic_type_df = pd.DataFrame(data.iloc[1:1018])

kfold = KFold(n_splits=2, shuffle=True, random_state=42)

# Calculate cross-validation scores
meta_cv_scores = cross_val_score(meta_model, meta_features_train, y, cv=kfold, scoring='accuracy', n_jobs=-1)

# Predict using meta-features
meta_cv_pred = cross_val_predict(meta_model, meta_features_train, y, cv=kfold)

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

#joblib.dump(meta_model, 'st2_model.joblib')
#joblib.dump(meta_features_train, 'st2_features.joblib')
joblib.dump(metrics_df, 'st2_metrics.joblib')
joblib.dump(predicted_traffic_type_df, 'st2_traffic_type.joblib')