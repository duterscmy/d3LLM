#!/usr/bin/env python3
"""
Merge multiple trajectory JSON files from a directory into a single dataset.
Usage: python merge.py input_path output_path
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datasets import Dataset


def merge_trajectory_files(input_dir, output_path):
    """
    Merge all trajectory JSON files in input_dir into a single HuggingFace dataset.
    
    Args:
        input_dir (str): Directory containing trajectory JSON files to merge
        output_path (str): Path to save the merged dataset
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return False
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return False
    
    # Find all JSON files in the input directory
    json_files = list(input_path.glob("*.json"))
    json_files.extend(input_path.glob("*.JSON"))  # Case-insensitive matching
    
    if not json_files:
        print(f"No JSON files found in '{input_dir}'")
        return False
    
    print(f"Found {len(json_files)} JSON files to merge:")
    for f in json_files:
        print(f"  - {f.name}")
    
    all_data = []
    total_samples = 0
    
    # Merge all JSON files
    for json_file in sorted(json_files):  # Sort for consistent order
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    # If file contains a list, extend all_data
                    all_data.extend(data)
                    samples_count = len(data)
                elif isinstance(data, dict):
                    # If file contains a dictionary, check if it's a single sample or a dataset dict
                    if all(key in data for key in ["idx", "question", "trajectory"]):
                        # If it looks like a dataset dict, convert to list of samples
                        samples_count = len(data["idx"])
                        for i in range(samples_count):
                            sample = {
                                "idx": data["idx"][i],
                                "question": data["question"][i],
                                "prompt_ids": data["prompt_ids"][i],
                                "trajectory": data["trajectory"][i],
                                "final_output": data["final_output"][i],
                                "generated_text": data["generated_text"][i],
                                "llm_answer": data["llm_answer"][i],
                                "gt_answer": data["gt_answer"][i],
                                "is_correct": data["is_correct"][i],
                                "nfe": data.get("nfe", [0]*samples_count)[i],
                            }
                            all_data.append(sample)
                    else:
                        # Single sample dictionary
                        all_data.append(data)
                        samples_count = 1
                else:
                    print(f"Warning: {json_file.name} contains unsupported data type. Skipping.")
                    continue
                
                total_samples += samples_count
                print(f"Loaded {samples_count} samples from {json_file.name}")
                
        except json.JSONDecodeError as e:
            print(f"Error: {json_file.name} is not a valid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            return False
    
    if not all_data:
        print("No data loaded from any file")
        return False
    
    print(f"\nTotal samples loaded: {total_samples}")
    
    # Convert to dataset format with correctness check
    try:
        dataset_dict = {
            "idx": [d["idx"] for d in all_data],
            "question": [d["question"] for d in all_data],
            "prompt_ids": [d["prompt_ids"] for d in all_data],
            "trajectory": [d["trajectory"] for d in all_data],
            "final_output": [d["final_output"] for d in all_data],
            "generated_text": [d["generated_text"] for d in all_data],
            "llm_answer": [d["llm_answer"] for d in all_data],
            "gt_answer": [d["gt_answer"] for d in all_data],
            "is_correct": [d["is_correct"] for d in all_data],
            "nfe": [d.get("nfe", 0) for d in all_data],  # Add NFE field with default 0
        }
    except KeyError as e:
        print(f"Error: Missing key {e} in data. Please ensure all samples have the required fields.")
        print("Required fields: idx, question, prompt_ids, trajectory, final_output, generated_text, llm_answer, gt_answer, is_correct")
        return False
    
    # Print statistics
    num_correct = sum(dataset_dict["is_correct"])
    total = len(dataset_dict["is_correct"])
    accuracy = num_correct / total if total > 0 else 0
    avg_nfe = sum(dataset_dict["nfe"]) / total if total > 0 else 0
    
    print(f"\nStatistics:")
    print(f"  Correctness: {num_correct}/{total} = {accuracy:.2%}")
    print(f"  Average NFE: {avg_nfe:.2f}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as HuggingFace dataset
    try:
        final_dataset = Dataset.from_dict(dataset_dict)
        
        # Save to disk
        final_dataset.save_to_disk(str(output_dir))
        
        print(f"\nSuccessfully merged {len(json_files)} files")
        print(f"Total samples: {total_samples}")
        print(f"Dataset saved to: {output_path}")
        
        # Also save a JSON version for easy inspection
        json_output = output_dir.parent / f"{output_dir.name}.json"
        with open(json_output, 'w', encoding='utf-8') as f:
            # Save only first 5 samples to keep file size manageable
            json.dump(all_data[:5], f, indent=2, ensure_ascii=False)
        print(f"Sample JSON preview saved to: {json_output} (first 5 samples)")
        
        return True
        
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple trajectory JSON files into a single HuggingFace dataset"
    )
    parser.add_argument(
        "input_path",
        help="Directory containing trajectory JSON files to merge"
    )
    parser.add_argument(
        "output_path",
        help="Path to save the merged dataset (will be saved as HuggingFace dataset directory)"
    )
    
    args = parser.parse_args()
    
    success = merge_trajectory_files(args.input_path, args.output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()