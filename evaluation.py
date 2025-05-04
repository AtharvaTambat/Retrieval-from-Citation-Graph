# evaluation.py
import argparse
import pickle
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

from torch_geometric.nn import SAGEConv 

class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    @torch.no_grad()
    def encode_single_node(self, x):
        device = x.device
        empty_edge_index = torch.empty((2,0), dtype=torch.long, device=device)

        # Apply layers sequentially
        x = self.conv1(x, empty_edge_index).relu()
        x = self.conv2(x, empty_edge_index)
        return x

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' 
EMBEDDINGS_PATH = 'final_embeddings.pkl' 
TITLES_PATH = 'all_titles.pkl'
GNN_MODEL_PATH = 'gnn_model_state.pt' 

# --- GNN Dimensions (MUST match train_gnn.py) ---
GNN_IN_CHANNELS = 384 
GNN_HIDDEN_CHANNELS = 128 
GNN_OUT_CHANNELS = 128 

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

    ordered_titles = []
    embeddings_list = []
    for title in all_titles:
        if title in title_to_embedding:
             ordered_titles.append(title)
             embeddings_list.append(title_to_embedding[title])

    if not embeddings_list:
        print("Error: No embeddings found in the loaded file.")
        exit()

    embeddings_matrix = np.array(embeddings_list) 

    # Load GNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNLinkPredictor(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device)) 
    model.to(device) 
    model.eval() 

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

    ordered_titles, embeddings_matrix, gnn_model, device = load_artifacts()
    sentence_model = SentenceTransformer(MODEL_NAME)
    test_text = args.test_paper_title + " " + args.test_paper_abstract

    test_embedding_384_np = sentence_model.encode(test_text, convert_to_tensor=False)

    test_embedding_384_tensor = torch.tensor(test_embedding_384_np).float().unsqueeze(0).to(device)
    test_embedding_128_tensor = gnn_model.encode_single_node(test_embedding_384_tensor)
    test_embedding_128_np = test_embedding_128_tensor.cpu().numpy() 

    similarities = cosine_similarity(test_embedding_128_np, embeddings_matrix) 
    sim_scores = similarities[0]
    ranked_indices = np.argsort(sim_scores)[::-1]

    ranked_titles = [ordered_titles[i] for i in ranked_indices]

    result = ranked_titles

    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    for title in result:
        print(title)

if __name__ == "__main__":
    main()