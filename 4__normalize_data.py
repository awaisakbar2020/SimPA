import pandas as pd

def normalize_columns(row, columns):
    total = row[columns].sum()
    if total != 0:
        row[columns] = row[columns] / total
    return row

def normalize_csv(filepath, output_filepath):
    # Read the CSV file
    data = pd.read_csv(filepath)

    print("LENGTH OF DATA FRAME: ", len(data))

    # Define the column groups to be normalized
    column_groups = [
        ["d1_drive", "d1_pt", "d1_cycle", "d1_walk"],
        ["d2_drive", "d2_pt", "d2_cycle", "d2_walk"],
        ["d3_drive", "d3_pt", "d3_cycle", "d3_walk"],
        ["d4_drive", "d4_pt", "d4_cycle", "d4_walk"],
        ["d5_drive", "d5_pt", "d5_cycle", "d5_walk"]
    ]

    # Normalize each column group
    for cols in column_groups:
        data[cols] = data[cols].apply(lambda row: row / row.sum() if row.sum() != 0 else row, axis=1)

    # Write the normalized data to a new CSV file
    data.to_csv(output_filepath, index=False)

    return data

normalized_data = normalize_csv("data/LTDS_cluster_centroids.csv", "data/LTDS_centroids_normalised_0_1.csv")
normalized_data = normalize_csv("data/LTDS_cluster_standard_deviation.csv", "data/LTDS_standard_deviation_normalised_0_1.csv")


print(normalized_data)
