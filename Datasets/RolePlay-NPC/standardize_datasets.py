
#!/usr/bin/env python3
"""
Script to standardize and combine JSONL datasets with different schemas.
Extracts only 'role' and 'content' fields to create a consistent schema.
"""

import json
import sys
import os
from typing import List, Dict, Any

def extract_standard_messages(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract messages in standard format from various schema structures.
    
    Args:
        data: A single JSONL record
        
    Returns:
        List of messages with only 'role' and 'content' fields
    """
    standard_messages = []
    
    # Handle your game_dataset format (has 'messages' array)
    if 'messages' in data and isinstance(data['messages'], list):
        for msg in data['messages']:
            if 'role' in msg and 'content' in msg:
                standard_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
    
    # Handle pippa_chatml format (also has 'messages' array but with extra fields)
    elif 'messages' in data and isinstance(data['messages'], list):
        for msg in data['messages']:
            if 'role' in msg and 'content' in msg:
                standard_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
    
    # Handle single message format (if the record itself is a message)
    elif 'role' in data and 'content' in data:
        standard_messages.append({
            'role': data['role'],
            'content': data['content']
        })
    
    return standard_messages

def standardize_jsonl_file(input_file: str, output_file: str) -> int:
    """
    Standardize a JSONL file to contain only conversations with role/content format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output standardized JSONL file
        
    Returns:
        Number of conversations processed
    """
    conversations_processed = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    messages = extract_standard_messages(data)
                    
                    if messages:
                        # Create standardized conversation record
                        standardized_record = {"messages": messages}
                        json.dump(standardized_record, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        conversations_processed += 1
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num} in {input_file}: {e}")
                    continue
                    
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} lines from {input_file}")
                    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return 0
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return 0
    
    return conversations_processed

def combine_standardized_files(file_paths: List[str], output_file: str) -> int:
    """
    Combine multiple standardized JSONL files into one.
    
    Args:
        file_paths: List of input file paths
        output_file: Path to combined output file
        
    Returns:
        Total number of conversations combined
    """
    total_conversations = 0
    
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    print(f"Warning: File '{file_path}' not found, skipping")
                    continue
                
                print(f"Adding conversations from {file_path}")
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()
                        if line:
                            outfile.write(line + '\n')
                            total_conversations += 1
                            
    except Exception as e:
        print(f"Error combining files: {e}")
        return 0
    
    return total_conversations

def main():
    # File paths
    game_dataset = "game_dataset.jsonl"
    pippa_dataset = "pippa_chatml.jsonl"
    
    # Standardized output files
    game_standardized = "game_dataset_standardized.jsonl"
    pippa_standardized = "pippa_chatml_standardized.jsonl"
    
    # Combined output file
    combined_output = "combined_dataset.jsonl"
    
    print("Starting dataset standardization...")
    
    # Standardize game_dataset.jsonl
    if os.path.exists(game_dataset):
        print(f"\nStandardizing {game_dataset}...")
        game_count = standardize_jsonl_file(game_dataset, game_standardized)
        print(f"Processed {game_count} conversations from {game_dataset}")
    else:
        print(f"Warning: {game_dataset} not found")
        game_count = 0
    
    # Standardize pippa_chatml.jsonl
    if os.path.exists(pippa_dataset):
        print(f"\nStandardizing {pippa_dataset}...")
        pippa_count = standardize_jsonl_file(pippa_dataset, pippa_standardized)
        print(f"Processed {pippa_count} conversations from {pippa_dataset}")
    else:
        print(f"Warning: {pippa_dataset} not found")
        pippa_count = 0
    
    # Combine standardized files
    standardized_files = []
    if game_count > 0:
        standardized_files.append(game_standardized)
    if pippa_count > 0:
        standardized_files.append(pippa_standardized)
    
    if standardized_files:
        print(f"\nCombining standardized datasets...")
        total_count = combine_standardized_files(standardized_files, combined_output)
        print(f"\nCombined {total_count} conversations into {combined_output}")
        
        # Cleanup intermediate files
        for file_path in standardized_files:
            try:
                os.remove(file_path)
                print(f"Cleaned up {file_path}")
            except:
                pass
        
        print(f"\nDataset standardization complete!")
        print(f"Final dataset: {combined_output}")
        print(f"Total conversations: {total_count}")
        print(f"- Game dataset: {game_count}")
        print(f"- Pippa dataset: {pippa_count}")
        
    else:
        print("No datasets found to process")

if __name__ == "__main__":
    main()