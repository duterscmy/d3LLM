#!/usr/bin/env python3
"""
Merge multiple trajectory JSON files with format conversion and statistics.
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
    Merge all JSON files, convert format, and save as HuggingFace dataset.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return False
    
    # Find all JSON files
    json_files = list(input_path.glob("trajectory_part_*.json"))
    if not json_files:
        json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{input_dir}'")
        return False
    
    print(f"Found {len(json_files)} JSON files to merge:")
    for f in json_files:
        print(f"  - {f.name}")
    
    # Step 1: Load and extend all data
    all_data = []
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                    print(f"Loaded {len(data)} samples from {json_file.name}")
                else:
                    print(f"Warning: {json_file.name} is not a list, skipping")
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            return False
    
    if not all_data:
        print("No data loaded")
        return False
    
    print(f"\nTotal samples loaded: {len(all_data)}")
    
    # Step 2: Convert to dataset format with correctness check (exactly like your code)
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
        "nfe": [d.get("nfe", 0) for d in all_data],
    }
    
    # Step 3: Print statistics (exactly like your code)
    num_correct = sum(dataset_dict["is_correct"])
    total = len(dataset_dict["is_correct"])
    accuracy = num_correct / total if total > 0 else 0
    avg_nfe = sum(dataset_dict["nfe"]) / total if total > 0 else 0
    print(f"\nCorrectness: {num_correct}/{total} = {accuracy:.2%}")
    print(f"Average NFE: {avg_nfe:.2f}")
    
    # Step 4: Save as dataset
    # Step 4: Save as dataset - 分批处理避免OOM
    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 方法1：使用 Dataset.from_generator 分批创建
        def gen_samples():
            for sample in all_data:
                yield {
                    "idx": sample["idx"],
                    "question": sample["question"],
                    "prompt_ids": sample["prompt_ids"],
                    "trajectory": sample["trajectory"],
                    "final_output": sample["final_output"],
                    "generated_text": sample["generated_text"],
                    "llm_answer": sample["llm_answer"],
                    "gt_answer": sample["gt_answer"],
                    "is_correct": sample["is_correct"],
                    "nfe": sample.get("nfe", 0),
                }
        
        # 分批创建数据集
        final_dataset = Dataset.from_generator(gen_samples)
        final_dataset.save_to_disk(str(output_dir))
        
        print(f"\nSaved complete dataset with {len(all_data)} samples to {output_dir}")
        return True
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Merge trajectory JSON files into dataset")
    parser.add_argument("input_path", help="Directory containing JSON files")
    parser.add_argument("output_path", help="Path to save the merged dataset")
    args = parser.parse_args()
    
    success = merge_trajectory_files(args.input_path, args.output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()