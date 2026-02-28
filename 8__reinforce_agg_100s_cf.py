import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from policy_network import PolicyNetwork
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define environment for reward and corrective feedback generation
class TravelEnv:
    def __init__(self, data):
        self.data = data
        self.num_states = len(data)
    
    def get_state(self, idx):
        return torch.tensor(self.data[idx, :3], dtype=torch.float32, device=device)  # First 3 columns as input features
    
    def get_user_preference(self, idx):
        return np.argmax(self.data[idx, 3:])  # Last 4 columns as user preferences (highest probability)

    def reward_and_feedback(self, action, user_pref):
        reward = 1 if action == user_pref else -1
        corrective_feedback = user_pref if reward == -1 else None
        return reward, corrective_feedback

# REINFORCE algorithm with corrective feedback and early stopping
def reinforce(env, policy_network, optimizer, num_episodes=3000, gamma=0.99, patience=10, threshold=1e-2):
    episode_rewards = []
    episode_losses = []
    best_avg_reward = -float('inf')
    patience_counter = 0
    results = []

    for episode in range(num_episodes):
        log_probs = []
        rewards = []
        supervised_losses = []  # Collect supervised loss for rejected actions
        total_reward = 0

        for idx in range(env.num_states):
            state = env.get_state(idx)
            user_pref = env.get_user_preference(idx)
            action_probs = policy_network(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            # Reward and corrective feedback
            reward, feedback = env.reward_and_feedback(action.item(), user_pref)
            rewards.append(reward)
            total_reward += reward
            log_probs.append(action_dist.log_prob(action))

            # Supervised correction if action was wrong
            if feedback is not None:
                target = torch.tensor(feedback, dtype=torch.long, device=device)
                supervised_loss = torch.nn.functional.cross_entropy(action_probs.unsqueeze(0), target.unsqueeze(0))
                supervised_losses.append(supervised_loss)

        # Compute discounted returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Policy gradient loss
        policy_loss = [-log_prob * G for log_prob, G in zip(log_probs, returns)]
        total_policy_loss = torch.stack(policy_loss).sum()

        # Combine policy and supervised loss
        total_loss = total_policy_loss + torch.stack(supervised_losses).sum() if supervised_losses else total_policy_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        episode_rewards.append(total_reward)
        episode_losses.append(total_loss.item())

        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            avg_loss = sum(episode_losses[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Avg Reward: {avg_reward}, Avg Loss: {avg_loss}")
            
            results.append([episode + 1, total_reward, avg_reward, avg_loss])
            if avg_reward > best_avg_reward + threshold:
                best_avg_reward = avg_reward
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Converged after {episode + 1} episodes with Avg Reward: {best_avg_reward}")
                break

    return results, episode_rewards, episode_losses

def process_files():
    folder_path = 'data/alpha_u/train_a2/'
    user_ids_df = pd.read_csv('data/diverse_users.csv')
    user_ids = set(user_ids_df['user_id'].astype(str))
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.split('.')[0][1:] in user_ids]
    
    aggregated_results = []
    
    for file in files:
        print(f"Processing file: {file}")
        try:
            # Read only the first 100 rows
            df = pd.read_csv(os.path.join(folder_path, file), nrows=100)
            data = df.values
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        # Initialize environment and policy network
        env = TravelEnv(data)
        policy_network = PolicyNetwork(input_size=3, output_size=4).to(device)

        # Load pretrained model (if available)
        pretrained_model_path = "models/alpha1_c2_best_model.pth"
        try:
            policy_network.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print(f"Loaded pretrained model from {pretrained_model_path}")
        except FileNotFoundError:
            print(f"No pretrained model found at {pretrained_model_path}, starting from scratch.")

        # Initialize optimizer
        optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

        # Run the REINFORCE algorithm
        results, episode_rewards, episode_losses = reinforce(env, policy_network, optimizer, num_episodes=3000)

        # Append results for this file
        aggregated_results.append({
            'file': file,
            'results': results,
            'rewards': episode_rewards,
            'losses': episode_losses
        })

        # Save individual results for this file
        results_dir = "data/training/a2_100s"
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, f"results_{file}.csv")
        pd.DataFrame(results, columns=["Episode", "Total Reward", "Average Reward", "Average Loss"]).to_csv(result_file, index=False)

        models_dir = "models/a2_500s/"
        os.makedirs(models_dir, exist_ok=True)  # Ensure the directory exists
        torch.save(policy_network.state_dict(), f"{models_dir}trained_policy_{file}.pth")
        print(f"Policy network saved for {file}")

    return aggregated_results

if __name__ == "__main__":
    aggregated_results = process_files()
    # Visualize results if necessary
    for result in aggregated_results:
        plt.figure(figsize=(10, 5))
        plt.plot(result['rewards'], label='Rewards')
        plt.plot(result['losses'], label='Losses', linestyle='--')
        plt.title(f"Performance for {result['file']}")
        plt.xlabel("Episodes")
        plt.ylabel("Reward / Loss")
        plt.legend()
        plt.show()
