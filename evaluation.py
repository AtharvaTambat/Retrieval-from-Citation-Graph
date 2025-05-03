# evaluation.py
import argparse
import pickle
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- GNN Model Definition (Needs to be identical to train_gnn.py) ---
# You MUST have the model class definition available here
from torch_geometric.nn import SAGEConv # Import necessary layer

class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Ensure these dimensions match the ones used during training
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # This encode function is used during training with graph structure
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    # Add a method to process a single node's features without edges
    @torch.no_grad() # Ensure no gradients are computed
    def encode_single_node(self, x):
        # SAGEConv's forward pass handles nodes without neighbors
        # It essentially performs the linear transformation part.
        # We pass an empty edge_index.
        device = x.device
        empty_edge_index = torch.empty((2,0), dtype=torch.long, device=device)

        # Apply layers sequentially
        x = self.conv1(x, empty_edge_index).relu()
        x = self.conv2(x, empty_edge_index)
        return x

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' # MUST match the model used in training
EMBEDDINGS_PATH = 'final_embeddings.pkl' # These are the 128-dim GNN embeddings
TITLES_PATH = 'all_titles.pkl'
GNN_MODEL_PATH = 'gnn_model_state.pt' # Path to saved GNN weights
K_NEIGHBORS = 10 # Number of papers to return

# --- GNN Dimensions (MUST match train_gnn.py) ---
GNN_IN_CHANNELS = 384 # Sentence Transformer dimension
GNN_HIDDEN_CHANNELS = 128 # Hidden dimension used in GNNLinkPredictor
GNN_OUT_CHANNELS = 128 # Output dimension of GNNLinkPredictor

# --- Helper Functions ---
def load_artifacts():
    """Loads pre-computed embeddings, titles, and GNN model."""
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(TITLES_PATH) or not os.path.exists(GNN_MODEL_PATH):
        print(f"Error: Required files not found. Ensure '{EMBEDDINGS_PATH}', '{TITLES_PATH}', and '{GNN_MODEL_PATH}' exist.")
        print("Run train_gnn.py first.")
        exit()

    with open(EMBEDDINGS_PATH, 'rb') as f:
        title_to_embedding = pickle.load(f) # These are 128-dim
    with open(TITLES_PATH, 'rb') as f:
        all_titles = pickle.load(f)

    # Prepare embeddings matrix and corresponding titles list
    ordered_titles = []
    embeddings_list = []
    for title in all_titles:
        if title in title_to_embedding:
             ordered_titles.append(title)
             embeddings_list.append(title_to_embedding[title])

    if not embeddings_list:
        print("Error: No embeddings found in the loaded file.")
        exit()

    embeddings_matrix = np.array(embeddings_list) # Shape (N, 128)

    # Load GNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNLinkPredictor(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device)) # Load weights
    model.to(device) # Move model to appropriate device
    model.eval() # Set to evaluation mode

    return ordered_titles, embeddings_matrix, model, device

# --- Main Evaluation Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    args = parser.parse_args()

    ################################################
    #               YOUR CODE START                #
    ################################################

    # 1. Load necessary artifacts (including GNN model)
    ordered_titles, embeddings_matrix, gnn_model, device = load_artifacts()
    sentence_model = SentenceTransformer(MODEL_NAME) # Load sentence transformer

    # 2. Prepare the test paper's text
    test_text = args.test_paper_title + " " + args.test_paper_abstract

    # 3. Generate initial (384-dim) embedding for the test paper
    # Get numpy array first, then convert to tensor
    test_embedding_384_np = sentence_model.encode(test_text, convert_to_tensor=False)

    # Convert to tensor, add batch dimension (1, 384), and move to device
    test_embedding_384_tensor = torch.tensor(test_embedding_384_np).float().unsqueeze(0).to(device)

    # 4. Use the loaded GNN model to transform the test embedding to 128-dim
    # Use the new encode_single_node method
    test_embedding_128_tensor = gnn_model.encode_single_node(test_embedding_384_tensor)

    # Convert the 128-dim tensor back to a numpy array for cosine similarity
    test_embedding_128_np = test_embedding_128_tensor.cpu().numpy() # Shape (1, 128)

    # 5. Calculate cosine similarity between the GNN-processed test paper (128-dim)
    #    and all GNN-processed dataset papers (128-dim)
    # Now dimensions match: (1, 128) vs (N, 128)
    similarities = cosine_similarity(test_embedding_128_np, embeddings_matrix) # Shape: (1, N)

    # Get the similarities as a 1D array
    sim_scores = similarities[0]

    # 6. Rank dataset papers by similarity
    ranked_indices = np.argsort(sim_scores)[::-1]

    # 7. Get the titles of the top K papers
    ranked_titles = [ordered_titles[i] for i in ranked_indices]

    # Prepare the result list (top K or fewer)
    result = ranked_titles[:K_NEIGHBORS]

    ################################################
    #               YOUR CODE END                  #
    ################################################

    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    for title in result:
        print(title)

if __name__ == "__main__":
    main()