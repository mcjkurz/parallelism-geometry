#!/usr/bin/env python3
"""
Temporary script to convert all_couplets_final_predictions.xlsx to all_predictions.json
"""

import pandas as pd
import json
import os

def convert_xlsx_to_json():
    # Input and output file paths
    input_file = "data/all_couplets_final_predictions.xlsx"
    output_file = "data/all_predictions.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return False
    
    try:
        # Read the Excel file
        print(f"Reading {input_file}...")
        df = pd.read_excel(input_file)
        
        # Print column names to verify structure
        print("Columns found in Excel file:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        
        # Define the expected columns
        expected_columns = [
            'poet', 'line1', 'line2', 'ground_truth', 'bert_comm_predicted',
            'bert_no_comm_predicted', 'gpt4.1_predicted', 'ds_predicted',
            'gpt4.1_analysis', 'ds_reasoning', 'ds_analysis'
        ]
        
        # Check if all expected columns exist
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            print("Available columns:", list(df.columns))
        
        # Select only the expected columns that exist
        available_columns = [col for col in expected_columns if col in df.columns]
        df_filtered = df[available_columns]
        
        # Convert DataFrame to list of dictionaries
        print(f"Converting {len(df_filtered)} rows to JSON format...")
        records = df_filtered.to_dict('records')
        
        # Write to JSON file
        print(f"Writing to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully converted {len(records)} records to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

if __name__ == "__main__":
    success = convert_xlsx_to_json()
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
