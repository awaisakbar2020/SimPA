import pandas as pd
import numpy as np

data = pd.read_csv('data/generated_data_dirichlet.csv')

# Extract probability and age columns
probability_values = data['female']
age_values = data['age']

# Generate random numbers (same length as data)
random_numbers = np.random.rand(len(probability_values))

# Assign female (1) or male (0) based on probability
female = (probability_values > random_numbers).astype(int)

# Convert age to integer (assuming no decimals needed)
age_values = age_values.astype(int)  # Rounding might be needed

# Generate user IDs (assuming unique IDs are needed)
num_users = len(data)
user_ids = np.arange(1, num_users + 1)  # Start from 1

# Add the 'female' and (converted) 'age' columns
data['female'] = female
data['age'] = age_values
data['user_id'] = user_ids

# Save the DataFrame (optional)
data.to_csv('data/synthetic_users.csv', index=False)
