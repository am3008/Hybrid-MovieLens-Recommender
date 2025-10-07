# %%
# Import Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import csv
import os

print("Libraries imported.")

# %%
# Step 1: Load Datasets
# Define movie columns
movie_columns = [
    "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# Define genre columns
genre_columns = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# Load ratings data
ratings = pd.read_csv(
    "ml-100k/u.data",
    sep='\t',
    names=["user_id", "item_id", "rating", "timestamp"]
)

# Load movie metadata
movies = pd.read_csv(
    "ml-100k/u.item",
    sep='|',
    encoding="latin-1",
    header=None,
    names=movie_columns
)

# Convert 'movie_id' to numeric
movies['movie_id'] = pd.to_numeric(movies['movie_id'], errors='coerce')
movies = movies.dropna(subset=['movie_id'])
movies['movie_id'] = movies['movie_id'].astype(int)

# Load user data
users = pd.read_csv(
    "ml-100k/u.user",
    sep='|',
    names=["user_id", "age", "gender", "occupation", "zip_code"]
)

print("Datasets loaded.")

# %%
# Step 2: Merge Datasets
# Merge ratings with movies on 'item_id' and 'movie_id'
full_data = pd.merge(
    ratings, movies, left_on='item_id', right_on='movie_id'
)

# Merge the result with users on 'user_id'
full_data = pd.merge(full_data, users, on='user_id')

print("Data merged successfully.")

# %%
# Step 3: Data Preprocessing and Feature Engineering
# Encode 'gender' column
full_data['gender_encoded'] = full_data['gender'].map({'M': 0, 'F': 1})

# Encode 'occupation' column
le_occ = LabelEncoder()
full_data['occupation_encoded'] = le_occ.fit_transform(full_data['occupation'])

# Normalize 'age' column
scaler = StandardScaler()
full_data['age_normalized'] = scaler.fit_transform(full_data[['age']])

# Prepare user features
user_features = full_data[
    ['user_id', 'age_normalized', 'gender_encoded', 'occupation_encoded']
].drop_duplicates('user_id')

# Prepare item features
item_features = full_data[['movie_id'] + genre_columns].drop_duplicates('movie_id')

print("Data preprocessing and feature engineering completed.")

# %%
# Step 4: Split Data into Training and Testing Sets
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(
    splitter.split(full_data, groups=full_data['user_id'])
)
train_data = full_data.iloc[train_idx].copy()  # Ensure it's a copy
test_data = full_data.iloc[test_idx].copy()    # Ensure it's a copy

print("Data split into training and testing sets.")

# %%
# Step 5: Prepare Data for Modeling
# Create mappings from IDs to indices
user_ids = full_data['user_id'].unique()
item_ids = full_data['movie_id'].unique()

user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}

# Add index columns to training data using .loc
train_data.loc[:, 'user_idx'] = train_data['user_id'].map(user_id_to_idx)
train_data.loc[:, 'item_idx'] = train_data['movie_id'].map(item_id_to_idx)

# Prepare tensors for training
user_indices = torch.tensor(train_data['user_idx'].values, dtype=torch.long)
item_indices = torch.tensor(train_data['item_idx'].values, dtype=torch.long)
ratings_tensor = torch.tensor(train_data['rating'].values, dtype=torch.float32)

# Prepare user features tensor
user_feat_df = user_features.set_index('user_id').loc[
    train_data['user_id']
].reset_index(drop=True)
user_features_tensor = torch.tensor(
    user_feat_df[['age_normalized', 'gender_encoded', 'occupation_encoded']].values,
    dtype=torch.float32
)

# Prepare item features tensor
item_feat_df = item_features.set_index('movie_id').loc[
    train_data['movie_id']
].reset_index(drop=True)
item_features_tensor = torch.tensor(
    item_feat_df[genre_columns].values,
    dtype=torch.float32
)

print("Data prepared for modeling.")

# %%
# Step 6: Define the Hybrid Recommendation Model
class HybridRecommender(nn.Module):
    def __init__(self, n_users, n_items, n_factors,
                 n_user_features, n_item_features):
        super(HybridRecommender, self).__init__()
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        # Feature layers
        self.user_feat_fc = nn.Linear(n_user_features, n_factors)
        self.item_feat_fc = nn.Linear(n_item_features, n_factors)
        # Output layers
        self.fc1 = nn.Linear(n_factors * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, user_idx, item_idx, user_features, item_features):
        # Embeddings
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        # Feature embeddings
        user_feat_emb = self.user_feat_fc(user_features)
        item_feat_emb = self.item_feat_fc(item_features)
        # Combine embeddings
        user_vector = user_emb + user_feat_emb
        item_vector = item_emb + item_feat_emb
        # Concatenate user and item vectors
        x = torch.cat([user_vector, item_vector], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

print("Model defined.")

# %%
# Step 7: Train the Model
# Instantiate the model
n_users = len(user_ids)
n_items = len(item_ids)
n_user_features = user_features_tensor.shape[1]
n_item_features = item_features_tensor.shape[1]
n_factors = 20  # Number of latent factors

model = HybridRecommender(
    n_users, n_items, n_factors, n_user_features, n_item_features
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 5
batch_size = 1024

train_dataset = TensorDataset(
    user_indices, item_indices, user_features_tensor,
    item_features_tensor, ratings_tensor
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        user_idx_batch, item_idx_batch, user_feat_batch, \
        item_feat_batch, rating_batch = batch
        optimizer.zero_grad()
        predictions = model(
            user_idx_batch, item_idx_batch,
            user_feat_batch, item_feat_batch
        )
        loss = criterion(predictions, rating_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}')

print("Model trained.")

# %%
# Step 8: Evaluate the Model
# Prepare test data
test_data['user_idx'] = test_data['user_id'].map(user_id_to_idx)
test_data['item_idx'] = test_data['movie_id'].map(item_id_to_idx)

test_user_indices = torch.tensor(
    test_data['user_idx'].values, dtype=torch.long
)
test_item_indices = torch.tensor(
    test_data['item_idx'].values, dtype=torch.long
)

test_user_feat_df = user_features.set_index('user_id').loc[
    test_data['user_id']
].reset_index(drop=True)
test_user_features_tensor = torch.tensor(
    test_user_feat_df[['age_normalized', 'gender_encoded', 'occupation_encoded']].values,
    dtype=torch.float32
)

test_item_feat_df = item_features.set_index('movie_id').loc[
    test_data['movie_id']
].reset_index(drop=True)
test_item_features_tensor = torch.tensor(
    test_item_feat_df[genre_columns].values,
    dtype=torch.float32
)

test_ratings = test_data['rating'].values

# Evaluate the model
model.eval()
with torch.no_grad():
    test_predictions = model(
        test_user_indices, test_item_indices,
        test_user_features_tensor, test_item_features_tensor
    ).numpy()

mae = mean_absolute_error(test_ratings, test_predictions)
rmse = np.sqrt(mean_squared_error(test_ratings, test_predictions))
print(f'Test MAE: {mae}')
print(f'Test RMSE: {rmse}')

# %%
# Step 9: Define Functions for Generating Recommendations
def get_feature_importance(model):
    # Extract weights from the user and item feature layers
    user_feature_weights = model.user_feat_fc.weight.data.cpu().numpy()
    item_feature_weights = model.item_feat_fc.weight.data.cpu().numpy()
    return user_feature_weights, item_feature_weights

def generate_recommendations_with_explanations(user_id, n=10):
    """
    Generates Top-N movie recommendations for a user with explanations.

    Parameters:
    - user_id (int): The ID of the user.
    - n (int): The number of recommendations to generate.

    Returns:
    - recommendations (list): List of tuples (movie_id, title, explanation).
    - recommended_item_ids (list): List of recommended movie IDs.
    """
    model.eval()
    user_idx = torch.tensor([user_id_to_idx[user_id]], dtype=torch.long)
    # Get user features
    user_feat = user_features[user_features['user_id'] == user_id][
        ['age_normalized', 'gender_encoded', 'occupation_encoded']
    ].values
    user_feat_tensor = torch.tensor(user_feat, dtype=torch.float32)
    # Predict ratings for all items
    item_indices = torch.tensor(range(n_items), dtype=torch.long)
    # Get item features
    item_feat_tensor = torch.tensor(
        item_features[genre_columns].values,
        dtype=torch.float32
    )
    with torch.no_grad():
        predictions = model(
            user_idx.repeat(n_items),
            item_indices,
            user_feat_tensor.repeat(n_items, 1),
            item_feat_tensor
        )
    # Exclude items already rated
    rated_items = train_data[train_data['user_id'] == user_id]['item_idx'].tolist()
    predictions[rated_items] = -np.inf  # Exclude rated items
    # Get top N items
    top_n_indices = torch.topk(predictions, n).indices.numpy()
    recommended_item_ids = [item_ids[idx] for idx in top_n_indices]
    # Get explanations
    user_feature_weights, item_feature_weights = get_feature_importance(model)
    recommendations = []
    for idx in top_n_indices:
        item_id = item_ids[idx]
        item_title = movies[movies['movie_id'] == item_id]['title'].values[0]
        item_genres = item_features[item_features['movie_id'] == item_id][
            genre_columns
        ].values  # Shape: (1, n_item_features)
        # Sum the item feature weights over latent factors
        item_feature_weights_sum = item_feature_weights.sum(axis=0)  # Shape: (n_item_features,)
        # Calculate contribution from item features per genre
        item_contrib = item_genres[0] * item_feature_weights_sum  # Element-wise multiplication
        # Get top contributing genres
        top_genres_indices = np.argsort(-item_contrib)[:3]
        top_genres = [genre_columns[i] for i in top_genres_indices if item_contrib[i] != 0]
        if not top_genres:
            explanation = "Recommended based on similar users' preferences."
        else:
            explanation = f"Recommended because you like {', '.join(top_genres)} movies."
        recommendations.append((item_id, item_title, explanation))
    return recommendations, recommended_item_ids

print("Functions for generating recommendations defined.")

# %%
# Step 10: Define Function to Save Recommendations
def save_recommendations_to_file(user_id, recommendations, filename='recommendations.csv'):
    """
    Saves the recommendations to a CSV file.

    Parameters:
    - user_id (int): The ID of the user.
    - recommendations (list): List of tuples (movie_id, title, explanation).
    - filename (str): The name of the file to save recommendations.
    """
    file_exists = os.path.isfile(filename)
    # Open the file in append mode
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header only if file does not exist or is empty
        if not file_exists or os.path.getsize(filename) == 0:
            writer.writerow(['user_id', 'movie_id', 'movie_title', 'explanation'])
        # Write recommendations
        for movie_id, movie_title, explanation in recommendations:
            writer.writerow([user_id, movie_id, movie_title, explanation])
    print(f"Recommendations for user {user_id} saved to {filename}.")

print("Function for saving recommendations defined.")

# %%
# Step 11: Generate Recommendations and Collect Data for Evaluation
# Clear the recommendations file before writing
open('recommendations.csv', 'w').close()

# Initialize dictionaries to store recommendations and testing items
user_recommendations = {}
user_testing_items = {}

test_user_ids = test_data['user_id'].unique()[:100]  # Adjust as needed

for user_id in test_user_ids:
    recommendations, recommended_item_ids = generate_recommendations_with_explanations(user_id)
    # Store recommended item IDs for evaluation
    user_recommendations[user_id] = set(recommended_item_ids)
    # Get testing items for the user
    testing_items = set(test_data[test_data['user_id'] == user_id]['movie_id'])
    user_testing_items[user_id] = testing_items
    # Save recommendations to file
    save_recommendations_to_file(user_id, recommendations, filename='recommendations.csv')

print("Recommendations generated and data collected for evaluation.")

# %%
# Step 12: Implement NDCG Function
def ndcg_at_k(recommended_items, testing_items, k=10):
    dcg = 0.0
    idcg = 0.0
    recommended_items = list(recommended_items)[:k]
    testing_items_list = list(testing_items)
    for i, item in enumerate(recommended_items):
        if item in testing_items:
            dcg += 1 / np.log2(i + 2)  # i + 2 because index i starts at 0
    # Ideal DCG (IDCG)
    for i in range(min(len(testing_items_list), k)):
        idcg += 1 / np.log2(i + 2)
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

print("NDCG function implemented.")

# %%
# Step 13: Compute Evaluation Metrics
precision_list = []
recall_list = []
f_measure_list = []
ndcg_list = []

for user_id in test_user_ids:
    recommended_items = user_recommendations[user_id]
    testing_items = user_testing_items[user_id]
    num_recommended = len(recommended_items)
    num_testing = len(testing_items)
    # Number of relevant items recommended
    relevant_recommended = recommended_items.intersection(testing_items)
    num_relevant_recommended = len(relevant_recommended)
    # Precision
    precision = num_relevant_recommended / num_recommended if num_recommended > 0 else 0
    # Recall
    recall = num_relevant_recommended / num_testing if num_testing > 0 else 0
    # F-measure
    if precision + recall > 0:
        f_measure = 2 * precision * recall / (precision + recall)
    else:
        f_measure = 0
    # NDCG
    ndcg = ndcg_at_k(recommended_items, testing_items, k=10)
    # Append to lists
    precision_list.append(precision)
    recall_list.append(recall)
    f_measure_list.append(f_measure)
    ndcg_list.append(ndcg)

# Calculate average metrics
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f_measure = np.mean(f_measure_list)
avg_ndcg = np.mean(ndcg_list)

print(f'Average Precision: {avg_precision:.4f}')
print(f'Average Recall: {avg_recall:.4f}')
print(f'Average F-measure: {avg_f_measure:.4f}')
print(f'Average NDCG: {avg_ndcg:.4f}')

# %%
# Step 14: Example of Displaying Recommendations for One User (Optional)
# You can display recommendations for a specific user
user_id = test_data['user_id'].iloc[0]  # Replace with a specific user ID if desired
recommendations, _ = generate_recommendations_with_explanations(user_id)
for movie_id, movie_title, explanation in recommendations:
    print(f"Recommended Movie: {movie_title} - {explanation}")

print("Example recommendations displayed.")
