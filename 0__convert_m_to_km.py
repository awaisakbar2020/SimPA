import pandas as pd

# Step 1: Load the data
file_path = 'data/LTDS_dataset_m.csv'  # Replace this with the path to your CSV file
data = pd.read_csv(file_path)

# Step 2: Convert distance from meters to kilometers
data['distance'] = data['distance'] / 1000  # Convert meters to kilometers

# Step 3: Save the updated DataFrame to a new CSV file
new_file_path = 'data/LTDS_dataset_km.csv'  # Replace this with your desired new file path
data.to_csv(new_file_path, index=False)

print(f"Data with distances in kilometers has been saved to: {new_file_path}")
