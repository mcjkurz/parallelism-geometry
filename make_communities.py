#!/usr/bin/env python3
"""
Character community detection for Chinese couplet parallelism analysis.
Builds a character co-occurrence network from parallel couplets and detects communities.
"""

import argparse
import json
import networkx as nx
from tqdm.auto import tqdm
from community import community_louvain


def load_training_data(input_file):
    """Load training data from JSON file and extract parallel couplets."""
    print(f"Loading training data from {input_file}...")
    
    with open(input_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    parallel_couplets = []
    for item in data:
        if item["label"] == 1:
            parallel_couplets.append((item["line1"], item["line2"]))
    
    print(f"Extracted {len(parallel_couplets)} parallel couplets for community detection")
    return parallel_couplets


def build_character_network(parallel_couplets, min_degree=None):
    """
    Build a character co-occurrence network from parallel couplets.
    Characters that appear in the same position across parallel lines are connected.
    """
    G = nx.Graph()

    print("Building character co-occurrence network...")
    for couplet in tqdm(parallel_couplets, desc="Processing couplets"):
        line1, line2 = couplet[0], couplet[1]
        
        for char1, char2 in zip(line1, line2):
            if char1 not in G:
                G.add_node(char1, label=char1)
            if char2 not in G:
                G.add_node(char2, label=char2)

            if G.has_edge(char1, char2):
                G[char1][char2]['weight'] += 1
            else:
                G.add_edge(char1, char2, weight=1)

    print(f"Initial network: {len(G.nodes())} nodes, {len(G.edges())} edges")

    if min_degree:
        print(f"Filtering nodes with degree < {min_degree}...")
        while True:
            nodes_to_remove = [node for node, degree in G.degree() if degree < min_degree]
            if not nodes_to_remove:
                break
            G.remove_nodes_from(nodes_to_remove)
            print(f"Nodes remaining: {len(G.nodes())}")

    print(f"Final network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return G


def detect_communities(G):
    """Detect communities using Louvain algorithm."""
    print("Detecting communities using Louvain algorithm...")
    partition = community_louvain.best_partition(G, weight='weight')
    
    nx.set_node_attributes(G, partition, name='community')
    
    num_communities = len(set(partition.values()))
    print(f"Detected {num_communities} communities")
    
    return G, partition


def save_communities(communities, output_file):
    """Save character communities to JSON file."""
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(communities, f, ensure_ascii=False, indent=2)
    
    print(f"Saved character communities to {output_file}")
    print(f"Total characters with community assignments: {len(communities)}")


def main():
    parser = argparse.ArgumentParser(description="Build character communities from parallel couplets")
    parser.add_argument("-i", "--input", required=True, 
                       help="Input JSON file with training data")
    parser.add_argument("-o", "--output", required=True, 
                       help="Output JSON file for communities")
    parser.add_argument("--min-degree", type=int, default=2,
                       help="Minimum degree for character nodes (default: 2)")
    
    args = parser.parse_args()
    
    parallel_couplets = load_training_data(args.input)
    
    network = build_character_network(parallel_couplets, min_degree=args.min_degree)
    
    network, communities = detect_communities(network)
    
    save_communities(communities, args.output)
    
    print("Community detection complete!")


if __name__ == "__main__":
    main()