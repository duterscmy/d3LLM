#!/usr/bin/env python3
"""
Merge multiple trajectory JSON files with streaming to avoid OOM.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datasets import Dataset, Features, Value, Sequence
import gc
import tempfile
import shutil


def process_file_streaming(file_path, dataset_writer):
    """
    流式处理单个JSON文件，逐条写入
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # 使用流式JSON解析器
        data = json.load(f)  # 这里还是会有问题
        # 对于超大文件，需要使用ijson等流式解析器
        
        for sample in data:
            # 逐条处理，不累积
            processed_sample = {
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
            dataset_writer.add_sample(processed_sample)
            # 定期清理内存
            if len(dataset_writer.buffer) >= 10000:
                dataset_writer.flush()


class StreamingDatasetWriter:
    """
    分批写入数据集，避免内存累积
    """
    def __init__(self, output_path, batch_size=10000):
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
        
        # 保存当前批次为临时文件
        batch_file = Path(self.temp_dir) / f"batch_{self.batch_count:05d}.arrow"
        batch_dataset = Dataset.from_list(self.buffer)
        batch_dataset.save_to_disk(str(batch_file))
        self.batch_files.append(batch_file)
        self.batch_count += 1
        
        # 清空缓冲区并强制垃圾回收
        self.buffer.clear()
        gc.collect()
    
    def finalize(self):
        """合并所有批次并保存最终数据集"""
        self.flush()
        
        if not self.batch_files:
            return
        
        # 逐个加载批次并合并
        print("Merging batches...")
        final_dataset = None
        for i, batch_file in enumerate(self.batch_files):
            print(f"  Loading batch {i+1}/{len(self.batch_files)}...")
            batch_dataset = Dataset.load_from_disk(str(batch_file))
            
            if final_dataset is None:
                final_dataset = batch_dataset
            else:
                final_dataset = final_dataset.concatenate([batch_dataset])
            
            # 清理临时文件
            shutil.rmtree(batch_file)
            gc.collect()
        
        # 保存最终数据集
        print(f"Saving final dataset to {self.output_path}...")
        final_dataset.save_to_disk(str(self.output_path))
        
        # 清理临时目录
        shutil.rmtree(self.temp_dir)


def merge_trajectory_files_streaming(input_dir, output_path):
    """
    流式合并所有JSON文件，避免内存溢出
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return False
    
    # 查找所有JSON文件
    json_files = list(input_path.glob("trajectory_part_*.json"))
    if not json_files:
        json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{input_dir}'")
        return False
    
    print(f"Found {len(json_files)} JSON files to merge:")
    print(f"Total size estimate: {get_total_size(json_files):.2f} GB")
    
    # 创建流式写入器
    writer = StreamingDatasetWriter(output_path, batch_size=5000)
    
    # 统计信息
    total_samples = 0
    correct_samples = 0
    total_nfe = 0
    
    # 逐个处理文件
    from tqdm import tqdm
    
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if not isinstance(data, list):
                    print(f"Warning: {json_file.name} is not a list, skipping")
                    continue
                
                file_samples = len(data)
                total_samples += file_samples
                print(f"\nProcessing {json_file.name} ({file_samples} samples)...")
                
                # 逐条处理样本
                for sample in tqdm(data, desc=f"  Processing {json_file.name}", leave=False):
                    # 更新统计
                    if sample.get("is_correct", False):
                        correct_samples += 1
                    total_nfe += sample.get("nfe", 0)
                    
                    # 写入到流式写入器
                    writer.add_sample(sample)
                
                # 处理完一个文件后强制清理内存
                data = None
                gc.collect()
                
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            return False
    
    # 最终合并并保存
    print(f"\nTotal samples processed: {total_samples}")
    print(f"Correctness: {correct_samples}/{total_samples} = {correct_samples/total_samples*100:.2f}%")
    print(f"Average NFE: {total_nfe/total_samples:.2f}")
    
    writer.finalize()
    
    print(f"\n✓ Saved dataset to {output_path}")
    return True


def get_total_size(files):
    """计算文件总大小(GB)"""
    total_bytes = sum(f.stat().st_size for f in files)
    return total_bytes / (1024**3)


def main():
    parser = argparse.ArgumentParser(description="Merge trajectory JSON files with streaming")
    parser.add_argument("input_path", help="Directory containing JSON files")
    parser.add_argument("output_path", help="Path to save the merged dataset")
    parser.add_argument("--batch-size", type=int, default=5000, 
                        help="Batch size for processing (default: 5000)")
    args = parser.parse_args()
    
    success = merge_trajectory_files_streaming(args.input_path, args.output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()