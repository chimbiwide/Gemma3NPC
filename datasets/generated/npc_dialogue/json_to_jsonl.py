#!/usr/bin/env python3
"""
Script to convert JSON array file to JSONL format.
Each object in the JSON array becomes a separate line in the JSONL file.
"""

import json
import sys
import os

def convert_json_to_jsonl(json_file_path, jsonl_file_path=None):
    """
    Convert a JSON array file to JSONL format.
    
    Args:
        json_file_path (str): Path to input JSON file
        jsonl_file_path (str): Path to output JSONL file (optional)
    """
    
    # If no output path specified, create one based on input path
    if jsonl_file_path is None:
        base_name = os.path.splitext(json_file_path)[0]
        jsonl_file_path = f"{base_name}.jsonl"
    
    try:
        # Read the JSON file
        print(f"Reading JSON file: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure data is a list
        if not isinstance(data, list):
            print("Error: JSON file must contain an array/list of objects")
            return False
        
        # Write to JSONL format
        print(f"Converting to JSONL format: {jsonl_file_path}")
        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Successfully converted {len(data)} objects")
        print(f"Output saved to: {jsonl_file_path}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    # Default file paths
    default_input = "game_dataset.json"
    default_output = "game_dataset.jsonl"
    
    # Check command line arguments
    if len(sys.argv) == 1:
        # No arguments, use defaults
        input_file = default_input
        output_file = default_output
    elif len(sys.argv) == 2:
        # One argument (input file)
        input_file = sys.argv[1]
        output_file = None  # Will be auto-generated
    elif len(sys.argv) == 3:
        # Two arguments (input and output files)
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        print("Usage:")
        print(f"  {sys.argv[0]}                    # Convert game_dataset.json to game_dataset.jsonl")
        print(f"  {sys.argv[0]} input.json         # Convert input.json to input.jsonl")
        print(f"  {sys.argv[0]} input.json output.jsonl  # Convert input.json to output.jsonl")
        return
    
    # Perform conversion
    success = convert_json_to_jsonl(input_file, output_file)
    
    if success:
        print("\nConversion completed successfully!")
    else:
        print("\nConversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()