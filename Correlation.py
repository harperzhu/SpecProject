import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# merged_data = pd.read_csv('DataSet/merged_pet_claims_data.csv')

# # Select only the numeric columns for correlation analysis
# # Update this list with the actual numeric columns from your dataset
# numeric_columns = ['PetAge', 'Premium', 'Breed', 'Species']
# data_subset = merged_data[numeric_columns]

# # Calculate the Pearson correlation matrix
# correlation_matrix = data_subset.corr()


# # Print the Pearson correlation matrix
# correlation_matrix.to_csv("Result/correlation.csv")


# # Generating pairplot for the selected variables
# sns.pairplot(data_subset)
# plt.show()



# Categorical variable
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load your dataset
merged_data = pd.read_csv('DataSet/merged_pet_claims_data.csv')

# ANOVA Example: Comparing 'Premium' across different 'Breeds'
# Ensure 'Breed' is a categorical variable in your dataset
anova_model = ols('Premium ~ C(Breed) + C(Species)', data=merged_data).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

relevant_variables= ['Premium','Breed','Species']
data_subset = merged_data[relevant_variables]


# Point-Biserial Correlation Example (assuming 'Species' is dichotomous)
# Convert 'Species' to a binary variable (0 and 1)
# Let's say 0 for 'Dog' and 1 for 'Cat'
merged_data['Species_binary'] = merged_data['Species'].map({'Dog': 0, 'Cat': 1})

# Calculate point-biserial correlation
from scipy.stats import pointbiserialr
correlation, p_value = pointbiserialr(merged_data['Species_binary'], merged_data['Premium'])
print("Correlation:", correlation, "P-value:", p_value)

# Save ANOVA table to CSV
anova_table.to_csv("Result/anova_table.csv")

# Save pairplot to image file
# sns.pairplot(data_subset)
# plt.savefig("Result/pairplot.png")

# Generate a scatter plot with contrasting colors for 'Dog' and 'Cat'
plt.figure(figsize=(10, 6))  # Adjust the size as needed
sns.scatterplot(data=merged_data, x='Breed', y='Premium', hue='Species', palette=['blue', 'orange'])

# Remove breed names from the x-axis
plt.xticks([])

# You can still keep the x-axis label if it's required for context
plt.xlabel('Breed Name')

# Show the legend and specify the title
plt.legend(title='Breed')

# Ensure layout fits the figure and labels
plt.tight_layout()

# Save the plot to a file
plt.savefig('cleaned_scatterplot.png')

plt.show()  # Show the plot