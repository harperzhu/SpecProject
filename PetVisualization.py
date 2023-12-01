import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

merged_data = pd.read_csv('DataSet/merged_pet_claims_data.csv')


# loading Data
cats_data = merged_data[merged_data['Species'] == 'Cat']
dogs_data = merged_data[merged_data['Species'] == 'Dog']


# Create the ordered list of age categories
age_order = ['<7 weeks', '<12 months'] + [str(i) for i in range(1, 14)]

# Calculate the average premium for each age group for both cats and dogs
age_premium_correlation_cat = cats_data.groupby('PetAge')['Premium'].mean().reset_index()
age_premium_correlation_dog = dogs_data.groupby('PetAge')['Premium'].mean().reset_index()

# Convert the 'PetAge' column to a categorical type with the specified order for sorting
age_premium_correlation_cat['PetAge'] = pd.Categorical(age_premium_correlation_cat['PetAge'], categories=age_order, ordered=True)
age_premium_correlation_dog['PetAge'] = pd.Categorical(age_premium_correlation_dog['PetAge'], categories=age_order, ordered=True)

# Sort the DataFrames
age_premium_correlation_cat.sort_values('PetAge', inplace=True)
age_premium_correlation_dog.sort_values('PetAge', inplace=True)

# Plot the line graphs for both cat and dog premiums
plt.figure(figsize=(12, 6))
sns.lineplot(data=age_premium_correlation_cat, x='PetAge', y='Premium', label='Cat Premium')
sns.lineplot(data=age_premium_correlation_dog, x='PetAge', y='Premium', label='Dog Premium')
plt.title('Correlation Between Pet Age and Premium for Cats and Dogs')
plt.xlabel('Pet Age')
plt.ylabel('Average Premium ($)')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('Result/cat_dog_age_premium_correlation.jpg')


# Time between claimDate adn EnrollmentDate for both cats and dogs
# Separate data for cats and dogs
cats_data = merged_data[merged_data['Species'] == 'Cat']
dogs_data = merged_data[merged_data['Species'] == 'Dog']

# Calculate the frequency of filing claims for each month duration for cats and dogs
claim_time_frequency_cat = cats_data['MonthsToClaim'].value_counts().sort_index()
claim_time_frequency_dog = dogs_data['MonthsToClaim'].value_counts().sort_index()

# Create the plot
plt.figure(figsize=(12, 6))

# Plotting the frequency of claim times for cats and dogs
sns.lineplot(x=claim_time_frequency_cat.index, y=claim_time_frequency_cat.values, label='Cats')
sns.lineplot(x=claim_time_frequency_dog.index, y=claim_time_frequency_dog.values, label='Dogs')

plt.title('Frequency of Claim Filing Time for Cats and Dogs')
plt.xlabel('Months to File a Claim')
plt.ylabel('Number of Samples')
plt.legend()
plt.show()
