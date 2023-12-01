from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  # Import the DecisionTreeRegressor
import numpy as np

# Load and preprocess data
merged_data = pd.read_csv('DataSet/merged_pet_claims_data.csv')

def process_age(age):
    if age == '0-7 weeks old' or age == '8 weeks to 12 months old':
        return 0

merged_data['PetAge'] = merged_data['PetAge'].apply(process_age)

# Handle categorical variables
label_encoder_species = LabelEncoder()
label_encoder_breed = LabelEncoder()
merged_data['Species'] = label_encoder_species.fit_transform(merged_data['Species'])
merged_data['Breed'] = label_encoder_breed.fit_transform(merged_data['Breed'])

# Define features and target
features = ['PetAge', 'Species', 'Breed']
target = 'Premium'

X = merged_data[features]
y = merged_data[target]

base_estimator = DecisionTreeRegressor(max_depth=10) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ada_reg = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=100, random_state=42)

# Train the model
ada_reg.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = ada_reg.predict(X_test)


# Perform cross-validation
cv_scores = cross_val_score(ada_reg, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert scores to positive values (since they are negative MSE)
mse_scores = -cv_scores

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")