import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import glob
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

def load_users_in_cluster(user_data_path, cluster_id):
    # Load user data to find users in the specified cluster
    user_data = pd.read_csv(user_data_path)
    
    # Filter users in the specified cluster
    cluster_users = user_data[user_data['cluster'] == cluster_id]['user_id'].tolist()
    
    return cluster_users

def load_training_data(folder_path, cluster_users):
    # Find all CSV files in the specified folder
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Filter the files to only include those belonging to users in the given cluster
    relevant_files = [file for file in all_files if int(os.path.basename(file).split('u')[1].split('.')[0]) in cluster_users]
    
    # Print the relevant files being loaded
    print(f"Loading the following files for training: {relevant_files}")
    
    # Load and concatenate the datasets from the relevant files
    df_list = [pd.read_csv(file) for file in relevant_files]
    full_data = pd.concat(df_list, ignore_index=True)
    
    return full_data

def train_policy_network(train_data_folder, user_data_path, cluster_id, model_save_path, patience=5):
    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Get the list of users in the specified cluster
    cluster_users = load_users_in_cluster(user_data_path, cluster_id)
    
    # Load the training data from files corresponding to the users in the specified cluster
    training_data = load_training_data(train_data_folder, cluster_users).values
    
    # Split into inputs (first 3 columns) and targets (probabilities)
    X_train = training_data[:, :3]  # First 3 columns as input features
    y_train = training_data[:, 3:]  # Probabilities for Walk, Cycle, Drive, Public Transport
    
    # Split the data into training and validation sets
    val_split = 0.2
    val_size = int(len(X_train) * val_split)
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    # Create DataLoaders for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Define model, loss, and optimizer
    input_dim = 3  # Updated to 3 since we have 3 input features now
    hidden_dim1 = 128
    hidden_dim2 = 64
    output_dim = 4
    policy_network = PolicyNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
    
    # Early stopping criteria
    best_val_loss = np.inf
    best_model_state = None
    patience_counter = 0
    num_epochs = 50
    
    for epoch in range(num_epochs):
        policy_network.train()
        running_loss = 0.0
        
        for inputs, probabilities in train_loader:
            optimizer.zero_grad()
            outputs = policy_network(inputs)
            loss = criterion(outputs, probabilities)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation step
        policy_network.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_probabilities in val_loader:
                val_outputs = policy_network(val_inputs)
                val_loss += criterion(val_outputs, val_probabilities).item()

        val_loss /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')
        
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = policy_network.state_dict()  # Save the best model state
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
        
        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
            break
    
    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print("Best model saved to", model_save_path)

if __name__ == "__main__":
    train_policy_network('data/alpha_u/train_a1/', 'data/synthetic_users.csv', 2, 'models/alpha1_c2_best_model.pth')
