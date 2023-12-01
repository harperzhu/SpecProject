import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load and preprocess data
merged_data = pd.read_csv('DataSet/merged_pet_claims_data.csv')

def process_age(age):
    if age == '0-7 weeks old' or age == '8 weeks to 12 months old':
        return 0

merged_data['PetAge'] = merged_data['PetAge'].apply(process_age)

# Assuming 'PetAge' is a numerical feature (modify according to your dataset)
numerical_features = ['PetAge']
categorical_features = ['Species', 'Breed']  # Assuming these are categorical


# Define your features and target variable
features = ['PetAge', 'Species', 'Breed']
target = 'Premium'

# Define your features and target variable
target = 'Premium'

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Prepare the data
X = merged_data[numerical_features + categorical_features]
y = merged_data[target]


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Convert to NumPy arrays if necessary
X_train = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

# Neural network model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=40, batch_size=32)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
# Check if y_test contains NaN values
print(f"NaN in y_test: {y_test.isna().any()}")

# # Since 'predictions' is a numpy array returned by the model, you can check for NaNs like this:
# print(f"NaN in predictions: {np.isnan(predictions).any()}")

# mse = mean_squared_error(y_test, predictions)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)

# print(f"Mean Squared Error: {mse}")
# print(f"Root Mean Squared Error: {rmse}")
# print(f"Mean Absolute Error: {mae}")
# print(f"R-squared: {r2}")