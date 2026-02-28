import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np

journeys_to_sample = 500
dataset_path = 'data/LTDS_dataset_km.csv'
alpha_u_model_path = 'data/diverse_users.csv'

def sample_diverse_journeys(journeys_to_sample, dataset_path, n_clusters=10):
    # Load the journey dataset
    journeys_df = pd.read_csv(dataset_path)
    
    # Extract distance values
    distances = journeys_df['distance'].values.reshape(-1, 1)
    
    # Apply KMeans clustering to ensure diverse samples
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(distances)
    
    # Create a DataFrame with distances and their cluster labels
    clustered_df = pd.DataFrame({
        'distance': journeys_df['distance'],
        'cluster': clusters
    })
    
    diverse_samples = []
    
    # Sample from each cluster to ensure diversity
    for cluster_label in range(n_clusters):
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_label]
        sample_size = min(journeys_to_sample // n_clusters, len(cluster_data))
        diverse_samples.append(cluster_data.sample(n=sample_size, random_state=42)['distance'])
    
    # Concatenate all the diverse samples and shuffle them
    diverse_samples = pd.concat(diverse_samples).sample(n=journeys_to_sample, random_state=42).reset_index(drop=True)
    
    # Split the sampled journeys into three parts
    first_third = diverse_samples[:journeys_to_sample].reset_index(drop=True)
    second_third = diverse_samples[journeys_to_sample:2*journeys_to_sample].reset_index(drop=True)
    third_third = diverse_samples[2*journeys_to_sample:].reset_index(drop=True)
    
    return first_third, second_third, third_third

# Sample the journeys with diversity
first_third_sampled_journeys, second_third_sampled_journeys, third_third_sampled_journeys = sample_diverse_journeys(journeys_to_sample, dataset_path)
print(f"Sampled {journeys_to_sample * 3} diverse distances.")

# Load the data from the CSV file
df = pd.read_csv(alpha_u_model_path)

def get_probabilities(distance, data):
    # Determine the distance category
    if 0.1 <= distance <= 2:
        category = 'd1'
    elif 2 < distance <= 4:
        category = 'd2'
    elif 4 < distance <= 6:
        category = 'd3'
    elif 6 < distance <= 8:
        category = 'd4'
    elif distance > 8:
        category = 'd5'
    else:
        print("Invalid distance")
        return None
    
    print(f"Distance: {distance} km falls into category: {category}")

    # Get the probabilities for the given category
    drive_prob = data[f'{category}_drive']
    pt_prob = data[f'{category}_pt']
    cycle_prob = data[f'{category}_cycle']
    walk_prob = data[f'{category}_walk']
    
    # Extract age and female columns as integers
    age = int(data['age'])  # Cast to integer
    female = int(data['female'])  # Cast to integer
    
    # Return the probabilities along with age and female as a dictionary
    probabilities = {
        'age': age,
        'female': female,
        'distance': distance,
        'drive_prob': drive_prob,
        'pt_prob': pt_prob,
        'cycle_prob': cycle_prob,
        'walk_prob': walk_prob
    }
    
    return probabilities

# Create directories if they don't exist
train_output_dir = 'data/alpha_u/train'
test_alpha_1_output_dir = 'data/alpha_u/test_alpha_1'
test_alpha_2_output_dir = 'data/alpha_u/test_alpha_2'
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_alpha_1_output_dir, exist_ok=True)
os.makedirs(test_alpha_2_output_dir, exist_ok=True)

# Process the first third sampled journeys for training
for i in range(len(df)):
    row = df.iloc[i]
    alpha_u_data = []

    # Predict mode of transport probabilities for each sampled journey
    for distance in first_third_sampled_journeys:
        probabilities = get_probabilities(distance, row)
        if probabilities:
            alpha_u_data.append(probabilities)

    # Convert the alpha_u_data to a DataFrame and save to CSV
    alpha_u_df = pd.DataFrame(alpha_u_data)
    alpha_u_df = alpha_u_df[['age', 'female', 'distance', 'drive_prob', 'pt_prob', 'cycle_prob', 'walk_prob']]  # Adjusted column order
    user_id = row['user_id']
    output_file_path = os.path.join(train_output_dir, f'u{user_id}.csv')
    alpha_u_df.to_csv(output_file_path, index=False)
    print(f'Saved predictions for user {user_id} to {output_file_path}')

# Similarly, process second and third thirds for testing
