import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from keras.optimizers import Adam  # Import the Adam optimizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.layers import Dense, BatchNormalization, Activation
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.regularizers import l2



# Load and preprocess data
merged_data = pd.read_csv('DataSet/merged_pet_claims_data.csv')
numerical_features = ['PetAge']
categorical_features = ['Species', 'Breed']  # Assuming these are categorical


# #check NaNs: 
# print("NaN in raw data:", merged_data.isna().any())
# merged_data['PetAge'] = merged_data['PetAge'].fillna(merged_data['PetAge'].median())

# merged_data['PetAge'] = merged_data['PetAge'].fillna(merged_data['PetAge'].median())

# for col in categorical_features:
#     merged_data[col] = merged_data[col].fillna('Unknown')


# Check if 'PetAge' column is entirely NaN or has some non-NaN values
print("Unique values in 'PetAge':", merged_data['PetAge'].unique())

# If it's not entirely NaN, fill NaNs with median or mean
if not merged_data['PetAge'].isna().all():
    median_age = merged_data['PetAge'].median()
    merged_data['PetAge'].fillna(median_age, inplace=True)
else:
    # If it's entirely NaN, you might need to reconsider how to handle this column
    print("'PetAge' column is entirely NaN.")


# Define your features and target variable
features = ['PetAge', 'Species', 'Breed']
target = 'AmountClaimed'

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

print("NaN in X_train:", np.isnan(X_train).any())
print("NaN in X_test:", np.isnan(X_test).any())

# Neural network model
model = Sequential()
# Input layer with batch normalization
model.add(Dense(128, input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Hidden layer with batch normalization
model.add(Dense(128, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.3))  # Adding dropout for regularization
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))

# Output layer
model.add(Dense(1))  # For regression tasks

# Instantiate the Adam optimizer with a lower learning rate
adam_optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)  # Add gradient clipping

# Compile the model with the new optimizer
model.compile(optimizer=adam_optimizer, loss='mean_squared_error')


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
# Check if y_test contains NaN values
# print(f"NaN in y_test: {X_test.np.isnan(predictions).any()}")

# Since 'predictions' is a numpy array returned by the model, you can check for NaNs like this:
print(f"NaN in predictions: {np.isnan(predictions).any()}")

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")