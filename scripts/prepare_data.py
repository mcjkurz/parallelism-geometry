#!/usr/bin/env python3
"""
Complete data preparation script for Chinese couplet parallelism detection.
Extracts couplets from CSV files and creates training datasets with optional community filtering.
"""

import argparse
import json
import re
import random
from pathlib import Path
from tqdm.auto import tqdm


def extract_couplets_from_files(data_dir, accepted_types=None, seed=42):
    """Extract couplets from CSV files."""
    if accepted_types is None:
        accepted_types = ["五言律诗"]
    
    random.seed(seed)
    
    files = [
        "唐.csv", "宋_1.csv", "宋_2.csv", "宋_3.csv", "元.csv",
        "明_1.csv", "明_2.csv", "明_3.csv", "明_4.csv",
        "清_1.csv", "清_2.csv", "清_3.csv"
    ]
    
    all_parallel_couplets = []
    all_nonparallel_couplets = []
    
    data_path = Path(data_dir)
    
    for file in tqdm(files, desc="Processing files"):
        file_path = data_path / file
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        with open(file_path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f.read().split("\n") if len(line.strip()) > 0]
            
        for line in tqdm(lines, desc=f"Processing {file}", leave=False):
            line_split = line.split(",")
            
            if not any(line_split.count(poem_type) for poem_type in accepted_types):
                continue
                
            poem = line_split[-1].strip()
            all_lines = [line for line in re.split(r"[。？，；！]", poem) 
                        if len(line) >= 1 and '□' not in line]
            
            if all(len(line) == 5 for line in all_lines) and len(all_lines) == 8:
                all_nonparallel_couplets.extend([
                    (all_lines[0], all_lines[1]), 
                    (all_lines[6], all_lines[7])
                ])
                all_parallel_couplets.extend([
                    (all_lines[2], all_lines[3]), 
                    (all_lines[4], all_lines[5])
                ])
    
    all_parallel_couplets = list(set(all_parallel_couplets))
    all_nonparallel_couplets = list(set(all_nonparallel_couplets))
    
    return all_parallel_couplets, all_nonparallel_couplets


def load_communities(communities_file):
    """Load character communities from JSON file."""
    try:
        with open(communities_file, "r", encoding='utf-8') as f:
            communities = json.load(f)
        return communities
    except FileNotFoundError:
        print(f"Communities file {communities_file} not found.")
        return None


def load_test_data(test_file):
    """Load test data and extract couplet pairs."""
    try:
        with open(test_file, "r", encoding='utf-8') as f:
            test_data = json.load(f)
        
        test_couplets = set()
        for item in test_data:
            couplet = (item["line1"], item["line2"])
            test_couplets.add(couplet)
            test_couplets.add((item["line2"], item["line1"]))
        
        return test_couplets
    except FileNotFoundError:
        print(f"Test file {test_file} not found.")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading test file {test_file}: {e}")
        return None


def remove_test_duplicates(parallel_couplets, nonparallel_couplets, test_couplets):
    """Remove any couplets from training data that appear in test data."""
    original_parallel_count = len(parallel_couplets)
    original_nonparallel_count = len(nonparallel_couplets)
    
    filtered_parallel = [couplet for couplet in parallel_couplets if couplet not in test_couplets]
    filtered_nonparallel = [couplet for couplet in nonparallel_couplets if couplet not in test_couplets]
    
    removed_parallel = original_parallel_count - len(filtered_parallel)
    removed_nonparallel = original_nonparallel_count - len(filtered_nonparallel)
    total_removed = removed_parallel + removed_nonparallel
    
    return filtered_parallel, filtered_nonparallel, total_removed


def filter_by_communities(parallel_couplets, nonparallel_couplets, communities, 
                         parallel_threshold=3, nonparallel_threshold=4):
    """
    Filter couplets based on character community analysis.
    
    Args:
        parallel_couplets: List of parallel couplets
        nonparallel_couplets: List of non-parallel couplets  
        communities: Dict mapping characters to community IDs
        parallel_threshold: Min same-community pairs for parallel couplets (default: 3)
        nonparallel_threshold: Min different-community pairs for non-parallel couplets (default: 4)
    
    Returns:
        Filtered parallel and non-parallel couplets, plus statistics
    """
    print(f"Filtering with thresholds: parallel≥{parallel_threshold} same, nonparallel≥{nonparallel_threshold} different")
    
    wrong_nonparallel = set()
    wrong_parallel = set()
    
    for cid, couplet in enumerate(tqdm(nonparallel_couplets, desc="Checking non-parallel")):
        line1, line2 = couplet
        same_community_count = 0
        
        for char1, char2 in zip(line1, line2):
            if char1 in communities and char2 in communities:
                if communities[char1] == communities[char2]:
                    same_community_count += 1
        
        if same_community_count >= parallel_threshold:
            wrong_nonparallel.add(cid)
    
    for cid, couplet in enumerate(tqdm(parallel_couplets, desc="Checking parallel")):
        line1, line2 = couplet
        different_community_count = 0
        
        for char1, char2 in zip(line1, line2):
            if char1 in communities and char2 in communities:
                if communities[char1] != communities[char2]:
                    different_community_count += 1
        
        if different_community_count >= nonparallel_threshold:
            wrong_parallel.add(cid)
    
    print(f"Found {len(wrong_parallel)} potentially mislabeled parallel couplets")
    print(f"Found {len(wrong_nonparallel)} potentially mislabeled non-parallel couplets")
    
    filtered_parallel = [couplet for cid, couplet in enumerate(parallel_couplets) 
                        if cid not in wrong_parallel]
    filtered_nonparallel = [couplet for cid, couplet in enumerate(nonparallel_couplets) 
                           if cid not in wrong_nonparallel]
    
    stats = {
        "original_parallel": len(parallel_couplets),
        "original_nonparallel": len(nonparallel_couplets),
        "filtered_parallel": len(filtered_parallel),
        "filtered_nonparallel": len(filtered_nonparallel),
        "removed_parallel": len(wrong_parallel),
        "removed_nonparallel": len(wrong_nonparallel)
    }
    
    return filtered_parallel, filtered_nonparallel, stats


def balance_dataset(parallel_couplets, nonparallel_couplets, seed=42):
    """Balance dataset to have equal numbers of parallel and non-parallel couplets."""
    random.seed(seed)
    
    min_count = min(len(parallel_couplets), len(nonparallel_couplets))
    print(f"Balancing dataset to {min_count} examples per class")
    
    balanced_parallel = random.sample(parallel_couplets, min_count)
    balanced_nonparallel = random.sample(nonparallel_couplets, min_count)
    
    return balanced_parallel, balanced_nonparallel


def create_train_json(parallel_couplets, nonparallel_couplets, output_file):
    """Create training dataset in standardized JSON format."""
    data = []
    
    for line1, line2 in parallel_couplets:
        data.append({
            "line1": line1,
            "line2": line2,
            "label": 1  # parallel
        })
    
    for line1, line2 in nonparallel_couplets:
        data.append({
            "line1": line1,
            "line2": line2,
            "label": 0  # non-parallel
        })
    
    random.shuffle(data)
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(data)} training examples to {output_file}")
    print(f"  - {len(parallel_couplets)} parallel couplets")
    print(f"  - {len(nonparallel_couplets)} non-parallel couplets")


def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset for couplet parallelism detection")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing CSV files")
    parser.add_argument("-c", "--communities", help="Character communities JSON file (enables community filtering)")
    parser.add_argument("-p", "--parallel-threshold", type=int, default=3,
                       help="Min same-community pairs for parallel couplets (default: 3)")
    parser.add_argument("-n", "--nonparallel-threshold", type=int, default=4,
                       help="Min different-community pairs for non-parallel couplets (default: 4)")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file for training dataset")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-t", "--test-data", help="Test data JSON file to exclude from training (e.g., data/test.json)")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("Extracting couplets from CSV files...")
    parallel_couplets, nonparallel_couplets = extract_couplets_from_files(
        data_dir=args.input_dir, 
        seed=args.seed
    )
    print(f"Extracted {len(parallel_couplets)} parallel and {len(nonparallel_couplets)} non-parallel couplets")
    
    if args.test_data:
        print(f"Loading test data from {args.test_data}...")
        test_couplets = load_test_data(args.test_data)
        
        if test_couplets is None:
            print("Cannot apply test data filtering without valid test file.")
            return
        
        print("Removing couplets that appear in test data...")
        parallel_couplets, nonparallel_couplets, removed_count = remove_test_duplicates(
            parallel_couplets, nonparallel_couplets, test_couplets
        )
        print(f"Removed {removed_count} couplets from training data (duplicated in test data)")
    else:
        print("No test data file provided, skipping test data filtering")
    
    if args.communities:
        print("Loading character communities...")
        communities = load_communities(args.communities)
        
        if communities is None:
            print("Cannot apply community filtering without valid communities file.")
            return
        
        print("Applying community-based filtering...")
        parallel_couplets, nonparallel_couplets, stats = filter_by_communities(
            parallel_couplets, nonparallel_couplets, communities,
            args.parallel_threshold, args.nonparallel_threshold
        )
    else:
        print("No communities file provided, skipping community-based filtering")
    
    print("Balancing dataset...")
    parallel_couplets, nonparallel_couplets = balance_dataset(
        parallel_couplets, nonparallel_couplets, args.seed
    )
    
    create_train_json(parallel_couplets, nonparallel_couplets, args.output)
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
