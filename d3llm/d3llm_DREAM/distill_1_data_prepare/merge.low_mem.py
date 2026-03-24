#!/usr/bin/env python3
"""
真正的流式合并，使用ijson避免一次性加载JSON
"""

import os
import sys
import json
import ijson
import argparse
from pathlib import Path
from datasets import Dataset, Features, Value, Sequence
import gc
import tempfile
import shutil
from tqdm import tqdm


class StreamingDatasetWriter:
    """分批写入数据集"""
    def __init__(self, output_path, batch_size=5000):
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.buffer = []
        self.batch_count = 0
        self.temp_dir = tempfile.mkdtemp()
        self.batch_files = []
        
    def add_sample(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        if not self.buffer:
            return
        
        batch_file = Path(self.temp_dir) / f"batch_{self.batch_count:05d}.arrow"
        batch_dataset = Dataset.from_list(self.buffer)
        batch_dataset.save_to_disk(str(batch_file))
        self.batch_files.append(batch_file)
        self.batch_count += 1
        
        self.buffer.clear()
        gc.collect()
    
    def finalize(self):
        """合并所有批次"""
        self.flush()
        
        if not self.batch_files:
            return
        
        print("Merging batches...")
        final_dataset = None
        for i, batch_file in enumerate(tqdm(self.batch_files, desc="Merging")):
            batch_dataset = Dataset.load_from_disk(str(batch_file))
            
            if final_dataset is None:
                final_dataset = batch_dataset
            else:
                final_dataset = final_dataset.concatenate([batch_dataset])
            
            shutil.rmtree(batch_file)
            gc.collect()
        
        print(f"Saving final dataset to {self.output_path}...")
        final_dataset.save_to_disk(str(self.output_path))
        shutil.rmtree(self.temp_dir)
        
        return final_dataset


def process_json_streaming(file_path, writer):
    """
    流式处理单个JSON文件，使用ijson避免一次性加载
    """
    samples_processed = 0
    
    with open(file_path, 'rb') as f:
        # 流式解析JSON数组中的每个元素
        parser = ijson.items(f, 'item')
        
        for sample in tqdm(parser, desc=f"  Processing {file_path.name}", leave=False):
            # 提取需要的字段
            processed_sample = {
                "idx": sample.get("idx"),
                "question": sample.get("question"),
                "prompt_ids": sample.get("prompt_ids"),
                "trajectory": sample.get("trajectory"),
                "final_output": sample.get("final_output"),
                "generated_text": sample.get("generated_text"),
                "llm_answer": sample.get("llm_answer"),
                "gt_answer": sample.get("gt_answer"),
                "is_correct": sample.get("is_correct", False),
                "nfe": sample.get("nfe", 0),
            }
            
            writer.add_sample(processed_sample)
            samples_processed += 1
            
            # 定期强制垃圾回收
            if samples_processed % 10000 == 0:
                gc.collect()
    
    return samples_processed


def merge_trajectory_files_streaming(input_dir, output_path, batch_size=5000):
    """
    流式合并所有JSON文件
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return False
    
    # 查找所有JSON文件
    json_files = sorted(list(input_path.glob("trajectory_part_*.json")))
    if not json_files:
        json_files = sorted(list(input_path.glob("*.json")))
    
    if not json_files:
        print(f"No JSON files found in '{input_dir}'")
        return False
    
    # 计算总大小
    total_size_gb = sum(f.stat().st_size for f in json_files) / (1024**3)
    print(f"Found {len(json_files)} JSON files to merge:")
    print(f"Total size estimate: {total_size_gb:.2f} GB")
    
    # 创建写入器
    writer = StreamingDatasetWriter(output_path, batch_size)
    
    # 统计信息
    total_samples = 0
    correct_samples = 0
    total_nfe = 0
    
    # 逐个处理文件
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            print(f"\nProcessing {json_file.name}...")
            
            file_samples = process_json_streaming(json_file, writer)
            total_samples += file_samples
            
            print(f"  Processed {file_samples} samples from {json_file.name}")
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 最终合并并保存
    print(f"\nTotal samples processed: {total_samples}")
    
    final_dataset = writer.finalize()
    
    # 计算最终统计（可选）
    if final_dataset is not None:
        correct_count = sum(final_dataset["is_correct"])
        avg_nfe = sum(final_dataset["nfe"]) / len(final_dataset)
        print(f"Correctness: {correct_count}/{total_samples} = {correct_count/total_samples*100:.2f}%")
        print(f"Average NFE: {avg_nfe:.2f}")
    
    print(f"\n✓ Saved dataset to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge trajectory JSON files with streaming")
    parser.add_argument("input_path", help="Directory containing JSON files")
    parser.add_argument("output_path", help="Path to save the merged dataset")
    parser.add_argument("--batch-size", type=int, default=5000, 
                        help="Batch size for processing (default: 5000)")
    args = parser.parse_args()
    
    success = merge_trajectory_files_streaming(args.input_path, args.output_path, args.batch_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()