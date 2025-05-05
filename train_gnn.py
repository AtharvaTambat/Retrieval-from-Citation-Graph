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


MODEL_NAME = 'all-MiniLM-L6-v2' 
EMBEDDING_DIM = 384 
OUT_CHANNELS = 128 
EPOCHS = 100000
LR = 0.01
K_NEIGHBORS = 100 

# --- Helper Functions ---
def load_paper_content(data_dir, graph_titles):
    """Loads title and abstract for papers present in the graph."""
    title_to_content = {}
    print(f"Loading content for {len(graph_titles)} papers...")
    
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
            if paper_title == title:
                title_to_content[title] = {'title': title, 'abstract': abstract}
         except FileNotFoundError:
             title_to_content[title] = {'title': title, 'abstract': ''} 
         except Exception as e:
            title_to_content[title] = {'title': title, 'abstract': ''}

    # Ensure all graph titles have an entry, even if content loading failed
    for title in graph_titles:
        if title not in title_to_content:
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

    edge_list = list(graph.edges())
    edge_index = torch.tensor([[node_map[u] for u, v in edge_list],
                               [node_map[v] for u, v in edge_list]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = len(nodes)
    data.node_map = node_map 
    data.nodes = nodes      

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
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
         prob_adj = z @ z.t()
         return prob_adj

# --- Training Function ---
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    pos_edge_index = data.train_pos_edge_index
    device = pos_edge_index.device 
    
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, 
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1), 
        method='sparse')
        
    neg_edge_index = neg_edge_index.to(device) 

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    

    edge_label = torch.cat([
        torch.ones(pos_edge_index.size(1), device=device), 
        torch.zeros(neg_edge_index.size(1), device=device)
    ], dim=0)

    z = model.encode(data.x, data.edge_index) 
    
    out = model.decode(z, edge_label_index) 
    
    loss = F.binary_cross_entropy_with_logits(out, edge_label) 
    
    loss.backward()
    optimizer.step()
    return loss.item(), z 

# --- Main Training Logic ---
def main_train(args):
    print("Loading graph...")
    with open(args.graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    graph_titles = set(G.nodes())
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
    
    for epoch in tqdm(range(1, EPOCHS + 1)):
        loss, z = train(model, data, optimizer)
        final_z = z 
        losses.append(loss) 
        if epoch % 500 == 0 or epoch == 1: 
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
    model.eval()
    gnn_model_filename = 'gnn_model_state.pt'
    torch.save(model.state_dict(), gnn_model_filename)
    print(f"Saved GNN model state to: {gnn_model_filename}")

    # --- Save Artifacts ---
    print("Saving trained artifacts...")

    model.eval()
    with torch.no_grad():
         final_z = model.encode(data.x, data.edge_index)

    final_embeddings_np = final_z.cpu().numpy()
    title_to_embedding = {title: final_embeddings_np[data.node_map[title]]
                          for title in data.nodes if title in data.node_map}

    with open('final_embeddings.pkl', 'wb') as f:
        pickle.dump(title_to_embedding, f)
    print("Saved final node embeddings to final_embeddings.pkl")

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