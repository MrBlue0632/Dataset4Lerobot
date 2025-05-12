#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成LerobotDataset格式的tasks.jsonl和episodes.jsonl文件。
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_total_episodes(input_dir):
    """检测数据集中的episode数量"""
    # 获取所有episode文件
    episode_files = list(input_dir.glob("lerobot_episode_*.parquet"))
    if not episode_files:
        print(f"错误: 在 {input_dir} 中未找到episode文件")
        return 0
        
    # 从文件名中提取episode编号
    episode_numbers = []
    for file in episode_files:
        try:
            # 从文件名中提取数字部分
            number = int(file.stem.split('_')[-1])
            episode_numbers.append(number)
        except ValueError:
            continue
            
    if not episode_numbers:
        print("错误: 无法从文件名中提取episode编号")
        return 0
        
    # 返回最大的episode编号
    total_episodes = max(episode_numbers)
    print(f"检测到 {total_episodes} 个episode")
    return total_episodes

def create_episodes_jsonl(output_dir, total_episodes, input_dir):
    """创建episodes.jsonl文件"""
    logger.info("开始生成episodes.jsonl...")
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for i in tqdm(range(total_episodes), desc="生成episodes"):
            # 读取原始数据文件以获取语言指导
            snapshot_file = input_dir / f"lerobot_episode_{i+1}.parquet"
            if not snapshot_file.exists():
                logger.warning(f"文件 {snapshot_file} 不存在，使用默认任务描述")
                language_instruction = "pick up the bottle and put it in the box"
            else:
                try:
                    # 读取原始数据文件
                    df = pd.read_parquet(snapshot_file)
                    logger.debug(f"读取文件 {snapshot_file} 的列: {df.columns.tolist()}")
                    
                    # 从数据中获取语言指导
                    if "language_instruction" in df.columns:
                        language_instruction = df["language_instruction"].iloc[0]
                        logger.debug(f"原始语言指令: {language_instruction}")
                        
                        # 如果语言指令为空，使用默认任务描述
                        if language_instruction is None:
                            language_instruction = "pick up the bottle and put it in the box"
                            logger.debug(f"使用默认语言指令: {language_instruction}")
                        # 处理可能的字节类型
                        elif isinstance(language_instruction, bytes):
                            language_instruction = language_instruction.decode('utf-8')
                            logger.debug(f"解码后的语言指令: {language_instruction}")
                        elif isinstance(language_instruction, str):
                            # 已经是字符串，直接使用
                            pass
                        else:
                            logger.warning(f"语言指令类型不是字符串或字节: {type(language_instruction)}")
                            language_instruction = "pick up the bottle and put it in the box"
                    else:
                        logger.warning(f"在文件 {snapshot_file} 中未找到language_instruction列")
                        language_instruction = "pick up the bottle and put it in the box"
                except Exception as e:
                    logger.error(f"读取语言指导时出错: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    language_instruction = "pick up the bottle and put it in the box"
            
            episode = {
                "episode_index": i,
                "tasks": [language_instruction],
                "length": len(df) if snapshot_file.exists() else 0
            }
            logger.debug(f"写入episode: {episode}")
            f.write(json.dumps(episode) + "\n")
    logger.info("episodes.jsonl生成完成")

def create_tasks_jsonl(output_dir, input_dir, total_episodes):
    """创建tasks.jsonl文件"""
    logger.info("开始生成tasks.jsonl...")
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        for i in tqdm(range(total_episodes), desc="生成tasks"):
            # 读取每个episode以获取语言指导
            snapshot_file = input_dir / f"lerobot_episode_{i+1}.parquet"
            if not snapshot_file.exists():
                logger.warning(f"文件 {snapshot_file} 不存在，使用默认任务描述")
                language_instruction = "pick up the bottle and put it in the box"
            else:
                try:
                    # 读取原始数据文件
                    df = pd.read_parquet(snapshot_file)
                    # 从数据中获取语言指导
                    if "language_instruction" in df.columns:
                        language_instruction = df["language_instruction"].iloc[0]
                        if isinstance(language_instruction, bytes):
                            language_instruction = language_instruction.decode('utf-8')
                        elif language_instruction is None:
                            language_instruction = "pick up the bottle and put it in the box"
                    else:
                        language_instruction = "pick up the bottle and put it in the box"
                except Exception as e:
                    logger.warning(f"读取语言指导时出错: {e}，使用默认任务描述")
                    language_instruction = "pick up the bottle and put it in the box"
            
            task = {
                "task_index": i,
                "task": language_instruction
            }
            f.write(json.dumps(task) + "\n")
    logger.info("tasks.jsonl生成完成")

def main():
    parser = argparse.ArgumentParser(description="生成LerobotDataset格式的tasks.jsonl和episodes.jsonl文件")
    parser.add_argument("--input_dir", type=str, default="/home/mrblue/dataset/xarm6", 
                        help="RLDS数据集目录路径")
    parser.add_argument("--output_dir", type=str, default="/home/mrblue/xarm6_lerobot_v2", 
                        help="LerobotDataset格式的输出目录")
    parser.add_argument("--debug", action="store_true", help="启用调试模式以显示详细日志")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 创建必要的目录
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    
    # 自动检测episode数量
    total_episodes = get_total_episodes(input_dir)
    if total_episodes == 0:
        return
    
    # 生成元数据文件
    create_episodes_jsonl(output_dir, total_episodes, input_dir)
    create_tasks_jsonl(output_dir, input_dir, total_episodes)
    
    logger.info(f"\n元数据文件生成完成！数据已保存到: {output_dir / 'meta'}")

if __name__ == "__main__":
    main() 