import pandas as pd
import os

journeys_to_sample = 500
dataset_path = 'data/LTDS_dataset_km.csv'
alpha_u_model_path = 'data/synthetic_users.csv'

def sample_journeys(journeys_to_sample, dataset_path):
    # Load the journey dataset
    journeys_df = pd.read_csv(dataset_path)
    
    # Sample three times the specified number of journeys
    total_samples = journeys_to_sample * 3
    sampled_journeys = journeys_df['distance'].sample(n=total_samples, random_state=42).reset_index(drop=True)
    
    # Split the sampled journeys into three parts
    first_third = sampled_journeys[:journeys_to_sample].reset_index(drop=True)
    second_third = sampled_journeys[journeys_to_sample:2*journeys_to_sample].reset_index(drop=True)
    third_third = sampled_journeys[2*journeys_to_sample:].reset_index(drop=True)
    
    return first_third, second_third, third_third

# Sample the journeys
first_third_sampled_journeys, second_third_sampled_journeys, third_third_sampled_journeys = sample_journeys(journeys_to_sample, dataset_path)
print(f"Sampled {journeys_to_sample * 3} distances.")

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
    
    # Print the extracted probabilities
    print(f"Probabilities for {category}:")
    print(f"Drive: {drive_prob}")
    print(f"PT: {pt_prob}")
    print(f"Cycle: {cycle_prob}")
    print(f"Walk: {walk_prob}")
    print(f"Age: {age}")
    print(f"Female: {female}")
    
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
train_output_dir = 'data/alpha_u/train_a1'
test_alpha_1_output_dir = 'data/alpha_u/train_a2'
test_alpha_2_output_dir = 'data/alpha_u/test_a1_a2'
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

# Process the second third sampled journeys for testing (test_alpha_1)
for i in range(len(df)):
    row = df.iloc[i]
    alpha_u_data = []

    # Predict mode of transport probabilities for each sampled journey
    for distance in second_third_sampled_journeys:
        probabilities = get_probabilities(distance, row)
        if probabilities:
            alpha_u_data.append(probabilities)

    # Convert the alpha_u_data to a DataFrame and save to CSV
    alpha_u_df = pd.DataFrame(alpha_u_data)
    alpha_u_df = alpha_u_df[['age', 'female', 'distance', 'drive_prob', 'pt_prob', 'cycle_prob', 'walk_prob']]  # Adjusted column order
    user_id = row['user_id']
    output_file_path = os.path.join(test_alpha_1_output_dir, f'u{user_id}.csv')
    alpha_u_df.to_csv(output_file_path, index=False)
    print(f'Saved predictions for user {user_id} to {output_file_path}')

# Process the third third sampled journeys for testing (test_alpha_2)
for i in range(len(df)):
    row = df.iloc[i]
    alpha_u_data = []

    # Predict mode of transport probabilities for each sampled journey
    for distance in third_third_sampled_journeys:
        probabilities = get_probabilities(distance, row)
        if probabilities:
            alpha_u_data.append(probabilities)

    # Convert the alpha_u_data to a DataFrame and save to CSV
    alpha_u_df = pd.DataFrame(alpha_u_data)
    alpha_u_df = alpha_u_df[['age', 'female', 'distance', 'drive_prob', 'pt_prob', 'cycle_prob', 'walk_prob']]  # Adjusted column order
    user_id = row['user_id']
    output_file_path = os.path.join(test_alpha_2_output_dir, f'u{user_id}.csv')
    alpha_u_df.to_csv(output_file_path, index=False)
    print(f'Saved predictions for user {user_id} to {output_file_path}')
