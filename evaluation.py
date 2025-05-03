# evaluation.py
import argparse
import pickle
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' # MUST match the model used in training
EMBEDDINGS_PATH = 'final_embeddings.pkl'
TITLES_PATH = 'all_titles.pkl'
K_NEIGHBORS = 100 # Number of papers to return

# --- Helper Functions ---
def load_artifacts():
    """Loads pre-computed embeddings and titles."""
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(TITLES_PATH):
        print(f"Error: Required files not found. Ensure '{EMBEDDINGS_PATH}' and '{TITLES_PATH}' exist.")
        print("Run train_gnn.py first.")
        exit()
        
    with open(EMBEDDINGS_PATH, 'rb') as f:
        title_to_embedding = pickle.load(f)
    with open(TITLES_PATH, 'rb') as f:
        all_titles = pickle.load(f)
    
    # Prepare embeddings matrix and corresponding titles list for faster similarity calculation
    ordered_titles = []
    embeddings_list = []
    for title in all_titles:
        if title in title_to_embedding: # Should always be true if created correctly
             ordered_titles.append(title)
             embeddings_list.append(title_to_embedding[title])
             
    if not embeddings_list:
        print("Error: No embeddings found in the loaded file.")
        exit()
        
    embeddings_matrix = np.array(embeddings_list)
    
    return ordered_titles, embeddings_matrix

# --- Main Evaluation Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    args = parser.parse_args()

    ################################################
    #               YOUR CODE START                #
    ################################################

    # 1. Load necessary artifacts
    ordered_titles, embeddings_matrix = load_artifacts()
    sentence_model = SentenceTransformer(MODEL_NAME) # Load the same sentence transformer

    # 2. Prepare the test paper's text
    test_text = args.test_paper_title + " " + args.test_paper_abstract

    # 3. Generate embedding for the test paper
    with torch.no_grad():
        test_embedding = sentence_model.encode(test_text, convert_to_tensor=False) # Get numpy array

    # Reshape for cosine similarity calculation (expects 2D arrays)
    test_embedding = test_embedding.reshape(1, -1) 

    # 4. Calculate cosine similarity between test paper and all dataset papers
    similarities = cosine_similarity(test_embedding, embeddings_matrix) # Shape: (1, num_dataset_papers)
    
    # Get the similarities as a 1D array
    sim_scores = similarities[0]

    # 5. Rank dataset papers by similarity
    # Get indices that would sort the similarities in descending order
    ranked_indices = np.argsort(sim_scores)[::-1]

    # 6. Get the titles of the top K papers
    ranked_titles = [ordered_titles[i] for i in ranked_indices]
    
    # Prepare the result list (top K or fewer if dataset is smaller)
    result = ranked_titles[:K_NEIGHBORS]

    ################################################
    #               YOUR CODE END                  #
    ################################################

    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    # Ensure only the ranked list is printed, one title per line
    for title in result:
        print(title)

if __name__ == "__main__":
    main()