import csv
import pandas as pd

# Define the filename
input_filename = "data/LTDS_dataset_processed.csv"
output_filename = "data/LTDS_dataset_refined.csv"

# Read the CSV file and refine the data
with open(input_filename, 'r') as file:
    reader = csv.DictReader(file)
    refined_data = []
    for row in reader:
        data_dict = eval(row['0'])
        refined_data.append(data_dict)

# Convert the list of dictionaries to a pandas DataFrame
refined_df = pd.DataFrame(refined_data)

# Save the refined data to a new CSV file
refined_df.to_csv(output_filename, index=False)

print(f"Refined data saved to: {output_filename}")
