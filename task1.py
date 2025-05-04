#!/usr/bin/env python3
import os
import glob
import re
import argparse

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle

def load_internal_titles(data_dir):
    """
    Load each paper's folder->title mapping and normalize titles.
    """
    folder_to_title = {}
    normalized_titles = {}
    for folder in os.listdir(data_dir):
        path = os.path.join(data_dir, folder)
        if not os.path.isdir(path):
            continue
        title_file = os.path.join(path, 'title.txt')
        try:
            with open(title_file, 'r', encoding='utf-8', errors='ignore') as f:
                title = f.read().strip()
        except FileNotFoundError:
            title = folder
        folder_to_title[folder] = title
        # Normalize: lowercase, strip punctuation
        norm = re.sub(r"[^a-z0-9 ]+", "", title.lower())
        normalized_titles[norm] = title
    return folder_to_title, normalized_titles


def parse_cited_titles(folder_path):
    """
    Parse .bib and .bbl in folder to return a set of cited titles (normalized).
    """
    cited_norm_titles = set()
    # .bib files
    for bib in glob.glob(os.path.join(folder_path, '*.bib')):
        try:
            content = open(bib, encoding='utf-8', errors='ignore').read()
        except Exception:
            continue
        # find title = { ... }
        for m in re.finditer(r"title\s*=\s*\{([^}]+)\}" , content, flags=re.IGNORECASE):
            title = m.group(1).strip()
            norm = re.sub(r"[^a-z0-9 ]+", "", title.lower())
            cited_norm_titles.add(norm)
    # .bbl files
    for bbl in glob.glob(os.path.join(folder_path, '*.bbl')):
        try:
            content = open(bbl, encoding='utf-8', errors='ignore').read()
        except Exception:
            continue
        # find each \bibitem block
        for m in re.finditer(r"\\bibitem(?:\[[^\]]*\])*\{[^}]+\}(?P<body>.*?)(?=(\\bibitem|$))", content, flags=re.DOTALL):
            body = m.group('body')
            parts = re.split(r"\\newblock", body)
            if len(parts) >= 2:
                title_part = parts[1].strip()
                # take text up to first period
                title = title_part.split('.',1)[0].strip()
                norm = re.sub(r"[^a-z0-9 ]+", "", title.lower())
                cited_norm_titles.add(norm)
    return cited_norm_titles


def build_graph(data_dir):
    folder_to_title, normalized_titles = load_internal_titles(data_dir)
    G = nx.DiGraph()

    # Add all internal titles as nodes
    for title in tqdm(folder_to_title.values(), desc="Adding nodes", unit="paper"):
        G.add_node(title)

    # For each paper, parse cited normalized titles and match
    for folder, src_title in tqdm(folder_to_title.items(), desc="Building edges", unit="paper"):
        cited_norms = parse_cited_titles(os.path.join(data_dir, folder))
        for norm in cited_norms:
            # If the cited title matches an internal title norm, add edge
            if norm in normalized_titles:
                tgt_title = normalized_titles[norm]
                G.add_edge(src_title, tgt_title)
    return G


def analyze_graph(G):
    num_edges = G.number_of_edges()
    isolated = [n for n, d in G.degree() if d == 0]
    num_isolated = len(isolated)

    avg_in = sum(d for _, d in G.in_degree()) / G.number_of_nodes()
    avg_out = sum(d for _, d in G.out_degree()) / G.number_of_nodes()

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {num_edges}")
    print(f"Number of isolated nodes: {num_isolated}")
    print(f"Average in-degree: {avg_in:.4f}")
    print(f"Average out-degree: {avg_out:.4f}")

    # Cap degrees at 100 and use bin width=1
    degrees = [min(d, 100) for _, d in G.degree()]
    bins = range(0, 101 + 1)  # 0 to 101 for bin width 1 up to cap
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=bins)
    plt.title('Degree Histogram (capped at 100)')
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig('degree_histogram.png')
    print("Saved degree histogram to: degree_histogram.png")

    with open('citation_graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    print("Saved graph to citation_graph.pkl")

    und = G.to_undirected()
    if und.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(und), key=len)
        sub = und.subgraph(largest_cc)
        try:
            diam = nx.diameter(sub)
            print(f"Diameter of largest component: {diam}")
        except nx.NetworkXError:
            print("Could not compute diameter (component too small/disconnected).")

def main():
    parser = argparse.ArgumentParser(description="Build and analyze citation graph (title-based)")
    parser.add_argument('--data_dir', required=True, help='Path to dataset_papers')
    args = parser.parse_args()
    print("Building citation graph...")
    G = build_graph(args.data_dir)
    print("Graph analysis:")
    analyze_graph(G)


if __name__ == '__main__':
    main()