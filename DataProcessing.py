import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the datasets
pet_data = pd.read_csv('DataSet/PetData.csv')
claims_data = pd.read_csv('DataSet/cleaned_claims.csv')

# Merge the datasets on 'PetId'
merged_data = pd.merge(pet_data, claims_data, on='PetId')

# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in merged_data.columns:
    merged_data = merged_data.drop(columns=['Unnamed: 0'])

# Process the 'PetAge' column
def process_age(age):
    if pd.isna(age):
        return np.nan  # Return NaN if the input is NaN
    elif age == '0-7 weeks old':
        return 0  # Example return value for '0-7 weeks old'
    elif age == '8 weeks to 12 months old':
        return 0.5  # Example return value for '8 weeks to 12 months old'
    else:
        # Extract the numeric part from the string and convert to an appropriate numeric value
        # Assumes the format is always '[number] year(s) old' for other cases
        return int(age.split()[0])


merged_data['PetAge'] = merged_data['PetAge'].apply(process_age)

# Convert 'ClaimDate' and 'EnrollDate_y' to datetime objects
merged_data['ClaimDate'] = pd.to_datetime(merged_data['ClaimDate'])
merged_data['EnrollDate_y'] = pd.to_datetime(merged_data['EnrollDate_y'])

# Define a function to calculate the difference in months
def month_diff(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

# Apply the function to calculate the difference in months
merged_data['MonthsToClaim'] = merged_data.apply(lambda row: month_diff(row['ClaimDate'], row['EnrollDate_y']), axis=1)


# Save the updated DataFrame
merged_data.to_csv('DataSet/merged_pet_claims_data.csv', index=False)


# nan_in_data = merged_data.isna().any()

# print(nan_in_data)
# print("Unique values in 'PetAge':", merged_data['PetAge'].unique())


# print(merged_data['PetAge'])


