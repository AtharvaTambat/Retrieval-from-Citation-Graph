# train_gnn.py
import os
import pickle
import argparse
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected, negative_sampling, train_test_split_edges
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt 

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' # Or another suitable model
EMBEDDING_DIM = 384 # Dimension of the chosen Sentence Transformer model
OUT_CHANNELS = 128 # GNN output embedding dimension
EPOCHS = 1000 # Adjust as needed
LR = 0.01
K_NEIGHBORS = 100 # Number of papers to return in evaluation

# --- Helper Functions ---
def load_paper_content(data_dir, graph_titles):
    """Loads title and abstract for papers present in the graph."""
    title_to_content = {}
    print(f"Loading content for {len(graph_titles)} papers...")
    # This assumes your data_dir contains folders named like 'paper_0', 'paper_1', etc.
    # and inside each, there's title.txt and abstract.txt
    # You might need to adapt this based on your exact data structure from Task 1
    
    # First, build a map from title back to folder (if needed)
    # This is inefficient, ideally load_internal_titles from Task 1 would return this map
    # For now, let's assume we can iterate through folders and check titles
    
    title_to_folder = {}
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, 'title.txt'), 'r', encoding='utf-8') as f:
                    title = f.read().strip()
                if title in graph_titles:
                     title_to_folder[title] = folder_path
            except FileNotFoundError:
                continue
                
    print(f"Found folders for {len(title_to_folder)} graph titles.")

    for title, folder_path in tqdm(title_to_folder.items(), desc="Loading content"):
         try:
            with open(os.path.join(folder_path, 'title.txt'), 'r', encoding='utf-8') as f:
                 paper_title = f.read().strip()
            with open(os.path.join(folder_path, 'abstract.txt'), 'r', encoding='utf-8') as f:
                 abstract = f.read().strip()
            # Ensure the title matches exactly, handling potential loading discrepancies
            if paper_title == title:
                title_to_content[title] = {'title': title, 'abstract': abstract}
            # else:
            #     print(f"Warning: Title mismatch for folder {folder_path}. Expected '{title}', found '{paper_title}'")
         except FileNotFoundError:
             # print(f"Warning: Missing title or abstract for {title} in {folder_path}")
             # Assign empty content if missing, GNN needs features for all nodes
             title_to_content[title] = {'title': title, 'abstract': ''} 
         except Exception as e:
            # print(f"Error loading content for {title} in {folder_path}: {e}")
            title_to_content[title] = {'title': title, 'abstract': ''}

    # Ensure all graph titles have an entry, even if content loading failed
    for title in graph_titles:
        if title not in title_to_content:
            # print(f"Warning: Content not found for graph title: {title}. Using empty content.")
            title_to_content[title] = {'title': title, 'abstract': ''}
            
    print(f"Loaded content for {len(title_to_content)} papers.")
    return title_to_content


def create_pyg_data(graph, title_to_content, sentence_model):
    """Creates PyTorch Geometric data object."""
    nodes = list(graph.nodes())
    node_map = {node: i for i, node in enumerate(nodes)}
    
    print("Generating initial node features...")
    features = []
    for node in tqdm(nodes, desc="Encoding papers"):
        content = title_to_content.get(node, {'title': node, 'abstract': ''})
        text = content['title'] + " " + content['abstract']
        with torch.no_grad():
            embedding = sentence_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        features.append(embedding)
    
    x = torch.stack(features).float()

    # Ensure edges use the integer mapping
    edge_list = list(graph.edges())
    edge_index = torch.tensor([[node_map[u] for u, v in edge_list],
                               [node_map[v] for u, v in edge_list]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = len(nodes)
    data.node_map = node_map # Store mapping
    data.nodes = nodes       # Store original node titles/ids
    
    # Convert to undirected for message passing if needed by GNN, but keep original edges for link prediction task
    # data.edge_index = to_undirected(data.edge_index) # Optional: Depends on GNN choice

    return data

# --- GNN Model ---
class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # Dot product decoder
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
         # Used for inference - compute scores for all pairs (inefficient)
         prob_adj = z @ z.t()
         return prob_adj

# --- Training Function ---
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    # Use existing edges as positive examples (already on the correct device if data is)
    pos_edge_index = data.train_pos_edge_index
    device = pos_edge_index.device # Get the device from data tensors
    
    # Sample negative edges - negative_sampling usually returns CPU tensors
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, # Sample based on all edges
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1), # Match number of positive samples
        method='sparse')
        
    # Move negative edges to the correct device
    neg_edge_index = neg_edge_index.to(device) 

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    
    # Create edge_label on the CORRECT device
    edge_label = torch.cat([
        torch.ones(pos_edge_index.size(1), device=device), # Specify device
        torch.zeros(neg_edge_index.size(1), device=device) # Specify device
    ], dim=0)

    z = model.encode(data.x, data.edge_index) # Use all edges for message passing
    
    # Pass edge_label_index (which is now on the correct device)
    out = model.decode(z, edge_label_index) 
    
    # Now both 'out' and 'edge_label' should be on the same device
    loss = F.binary_cross_entropy_with_logits(out, edge_label) 
    
    loss.backward()
    optimizer.step()
    return loss.item(), z # Return embeddings from this epoch

# --- Main Training Logic ---
def main_train(args):
    print("Loading graph...")
    with open(args.graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    graph_titles = set(G.nodes())

    # Load content only for nodes remaining in the graph
    title_to_content = load_paper_content(args.data_dir, graph_titles)

    print(f"Loading Sentence Transformer: {MODEL_NAME}")
    sentence_model = SentenceTransformer(MODEL_NAME)

    print("Creating PyG data object...")
    data = create_pyg_data(G, title_to_content, sentence_model)

    # Use the same device for model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = data.to(device)

    # --- Prepare for Link Prediction Training ---
    data.train_pos_edge_index = data.edge_index


    model = GNNLinkPredictor(EMBEDDING_DIM, 128, OUT_CHANNELS).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    print("Starting GNN training...")
    final_z = None
    losses = [] # <-- Initialize list to store losses
    
    for epoch in range(1, EPOCHS + 1):
        loss, z = train(model, data, optimizer)
        final_z = z # Keep track of the last embeddings
        losses.append(loss) # <-- Store loss for this epoch
        if epoch % 10 == 0 or epoch == 1: # Print first epoch and every 10th
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    print("Training finished.")

    # --- Plotting Loss ---
    print("Plotting training loss...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), losses, label='Training Loss')
    plt.title('GNN Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCEWithLogitsLoss)')
    plt.legend()
    plt.grid(True)
    loss_plot_filename = 'gnn_training_loss.png'
    plt.savefig(loss_plot_filename)
    print(f"Saved training loss plot to: {loss_plot_filename}")

    # --- Save GNN Model State ---
    model.eval() # Ensure model is in evaluation mode before saving
    gnn_model_filename = 'gnn_model_state.pt'
    torch.save(model.state_dict(), gnn_model_filename)
    print(f"Saved GNN model state to: {gnn_model_filename}")

    # --- Save Artifacts ---
    print("Saving trained artifacts...")

    # 1. Final Node Embeddings (using the model in eval mode)
    model.eval()
    with torch.no_grad():
         # Recompute final embeddings using the full graph structure after training
         final_z = model.encode(data.x, data.edge_index)

    final_embeddings_np = final_z.cpu().numpy()

    # Map embeddings back to original paper titles
    title_to_embedding = {title: final_embeddings_np[data.node_map[title]]
                          for title in data.nodes if title in data.node_map}

    with open('final_embeddings.pkl', 'wb') as f:
        pickle.dump(title_to_embedding, f)
    print("Saved final node embeddings to final_embeddings.pkl")

    # 2. List of all paper titles used in training (nodes in the final graph)
    all_titles = list(title_to_embedding.keys()) # Use titles that have embeddings
    with open('all_titles.pkl', 'wb') as f:
        pickle.dump(all_titles, f)
    print("Saved list of all paper titles to all_titles.pkl")

    print(f"Remember to use Sentence Transformer model '{MODEL_NAME}' during evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN for citation link prediction")
    parser.add_argument('--data_dir', required=True, help='Path to dataset_papers directory')
    parser.add_argument('--graph_path', default='citation_graph.pkl', help='Path to the saved NetworkX graph')
    args = parser.parse_args()
    main_train(args)