import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classify(data):
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

    # Define K-Fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Initialize KNN classifier
    knn_model = KNeighborsClassifier()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=kfold, scoring='accuracy')
    grid_search.fit(scaled_features, y)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Initialize KNN classifier with the best hyperparameters
    best_knn_model = KNeighborsClassifier(**best_params)

    # Fit the model
    best_knn_model.fit(scaled_features, y)

    # Make predictions
    y_pred = best_knn_model.predict(scaled_features)

    # Inverse transform predicted labels to original class labels
    predicted_traffic_type = label_encoder.inverse_transform(y_pred)

    # Assign predicted values to the entire DataFrame
    data['Predicted Traffic Type'] = predicted_traffic_type
    predicted_traffic_type_df = pd.DataFrame(data.iloc[1:1018])

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_knn_model, scaled_features, y, cv=kfold, scoring='accuracy')

    # Evaluate model performance
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    mean_cv_accuracy = cv_scores.mean()

    # Create a DataFrame with the metrics data
    metrics_data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "Cross-Val"],
    "Value": [accuracy, precision, recall, f1, mean_cv_accuracy]
    }
    metrics_df = pd.DataFrame(metrics_data)

    return predicted_traffic_type_df, metrics_df