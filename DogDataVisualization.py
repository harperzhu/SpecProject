import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

merged_data = pd.read_csv('DataSet/merged_pet_claims_data.csv')


## Dog data ## 
dogs_data = merged_data[merged_data['Species'] == 'Dog']

# Dogs: breed and premium
pivot_table = dogs_data.pivot_table(values='Premium', index='Breed', aggfunc='mean')
average_premium_by_breed = dogs_data.groupby('Breed')['Premium'].mean().reset_index()
average_premium_by_breed.to_csv('Result/average_premium_by_breed_dog.csv', index=False)
most_expensive_premium_dog_breeds = dogs_data.groupby('Breed')['Premium'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(14, 10)) 
sns.barplot(x=most_expensive_premium_dog_breeds.index, y=most_expensive_premium_dog_breeds.values, palette="mako")
plt.title('Top 10 Dog Breeds with Highest Average Premium', fontsize=16)
plt.xlabel('Dog Breed', fontsize=14)
plt.ylabel('Average Premium ($)', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')  
plt.tight_layout()
plt.savefig('Result/top_10_dog_breeds_premium.jpg')


# # Dogs: age and premium

age_premium_correlation = dogs_data.groupby('PetAge')['Premium'].mean().reset_index()
age_order = ['<7 weeks', '<12 months'] + [str(i) for i in range(1, 14)]
age_premium_correlation['PetAge'] = pd.Categorical(age_premium_correlation['PetAge'], categories=age_order, ordered=True)
age_premium_correlation.sort_values('PetAge', inplace=True)
plt.figure(figsize=(12, 6))
sns.lineplot(data=age_premium_correlation, x='PetAge', y='Premium')
plt.title('Correlation Between Dog Age and Premium')
plt.xlabel('Dog Age')
plt.ylabel('Average Premium ($)')
plt.xticks(rotation=45)
plt.savefig('Result/dog_age_premium_correlation.jpg')


# # Cats: age and annual premium and amount claimed and deductible 

age_premium_correlation = dogs_data.groupby('PetAge')['Premium'].mean().reset_index()
age_claim_correlation = merged_data.groupby('PetAge')['AmountClaimed'].mean().reset_index()
age_deductible_correlation = dogs_data.groupby('PetAge')['Deductible'].mean().reset_index()


age_order = ['<7 weeks', '<12 months'] + [str(i) for i in range(1, 14)]

age_premium_correlation['PetAge'] = pd.Categorical(age_premium_correlation['PetAge'], categories=age_order, ordered=True)
age_deductible_correlation['PetAge'] = pd.Categorical(age_deductible_correlation['PetAge'], categories=age_order, ordered=True)
age_claim_correlation['PetAge'] = pd.Categorical(age_claim_correlation['PetAge'], categories=age_order, ordered=True)
age_premium_correlation.sort_values('PetAge', inplace=True)

plt.figure(figsize=(12, 6))
sns.lineplot(data=age_premium_correlation, x='PetAge', y='Premium', label='Premium')
sns.lineplot(data=age_claim_correlation, x='PetAge', y='AmountClaimed', label='Amount Claimed')
sns.lineplot(data=age_deductible_correlation, x='PetAge', y='Deductible', label='Deductible')
plt.title('Correlation Between Dog Age, Premium, and Amount Claimed')
plt.xlabel('Dog Age')
plt.ylabel('Average Value ($)')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('Result/dog_age_premium_deductible_claim_amount_correlation.jpg')


