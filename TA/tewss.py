import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

uploaded_file = None

if uploaded_file is not None:
    # Read the uploaded file
    if uploaded_file.type == 'application/vnd.ms-excel' or uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        state = "new_excel"
    elif uploaded_file.type =='text/csv':
        state = "new_csv"
    else:
        st.write("File type is not supported")
else:
    state = "fixed_data"

        # Read the uploaded file
    if state == "new_excel":
        data = pd.read_excel(uploaded_file)
    elif state == "new_csv":
        data = pd.read_csv(uploaded_file)
    elif state == "fixed_data":
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
print("Shape of X before preprocessing:", X.shape)
print("Sample of X before preprocessing:")
print(X.head())
# Fit and transform the data
scaled_features = preprocessing_pipeline.fit_transform(X)
print("Shape of scaled_features:", scaled_features.shape)
print("Sample of scaled_features:")
print(scaled_features[:5])