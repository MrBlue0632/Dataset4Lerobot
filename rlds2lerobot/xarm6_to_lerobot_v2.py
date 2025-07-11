#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将RLDS格式的xarm6数据集转换为LerobotDataset v2.1格式。

使用方法:
1. 确保已安装所需依赖:
   pip install datasets pyarrow pandas numpy av opencv-python tqdm

2. 运行脚本:
   python convert_xarm6_data_to_lerobot.py --input_dir /path/to/xarm6_pick_bottle_in_box --output_dir /path/to/output
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import av
import cv2
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ===================== 数据读取相关函数 =====================

def get_snapshot_hash(input_dir):
    """从refs/main文件中读取snapshot哈希值"""
    ref_main_file = input_dir / "refs" / "main"
    if not ref_main_file.exists():
        raise FileNotFoundError(f"找不到refs/main文件: {ref_main_file}")
    
    with open(ref_main_file, 'r') as f:
        hash_value = f.read().strip()
    return hash_value

def get_total_episodes(input_dir):
    """检测数据集中的episode数量"""
    # 获取snapshot哈希值
    snapshot_hash = get_snapshot_hash(input_dir)
    snapshot_dir = input_dir / "snapshots" / snapshot_hash
    if not snapshot_dir.exists():
        print(f"错误: 目录 {snapshot_dir} 不存在")
        return 0
        
    # 获取所有episode文件
    episode_files = list(snapshot_dir.glob("lerobot_episode_*.parquet"))
    if not episode_files:
        print(f"错误: 在 {snapshot_dir} 中未找到episode文件")
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

# ===================== 数据处理相关函数 =====================

def process_image(image, width=640, height=480):
    """处理单个图像帧"""
    try:
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                # 调整图像大小
                img = cv2.resize(img, (width, height))
                return img
        elif isinstance(image, np.ndarray):
            # 调整图像大小
            img = cv2.resize(image, (width, height))
            return img
    except Exception as e:
        logger.debug(f"处理图像时出错: {e}")
    return None

def convert_episode(input_dir, output_dir, episode_index):
    """转换单个episode的数据"""
    try:
        logger.info(f"\n开始处理episode {episode_index}...")
        
        # 创建必要的目录
        data_chunk_dir = output_dir / "data" / "chunk-000"
        video_chunk_dir = output_dir / "videos" / "chunk-000"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)
        video_chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取snapshot哈希值
        snapshot_hash = get_snapshot_hash(input_dir)
        
        # 读取RLDS格式的数据
        snapshot_file = input_dir / "snapshots" / snapshot_hash / f"lerobot_episode_{episode_index+1}.parquet"
        logger.debug(f"读取文件: {snapshot_file}")
        df = pd.read_parquet(snapshot_file)
        
        # 创建LerobotDataset格式的DataFrame
        new_df = pd.DataFrame()
        
        # 转换图像数据为视频
        image_columns = [col for col in df.columns if 'image' in col.lower()]
        if image_columns:
            # 为每个图像列创建对应的视频目录
            for image_col in image_columns:
                # 创建视频目录（使用原数据集中的图像列名称）
                video_subdir = video_chunk_dir / f"observation.images.{image_col}"
                video_subdir.mkdir(parents=True, exist_ok=True)
                
                # 创建视频文件
                video_file = video_subdir / f"episode_{episode_index:06d}.mp4"
                logger.debug(f"创建视频文件: {video_file}")
                
                # 获取图像数据
                images = df[image_col].tolist()
                # 过滤掉None值
                images = [img for img in images if img is not None]
                if images:
                    create_video(images, video_file)
        else:
            logger.warning("警告: 未找到图像数据列")
        
        # 转换状态数据
        state_columns = [col for col in df.columns if 'state' in col.lower()]
        if state_columns:
            state_col = state_columns[0]
            logger.debug(f"使用状态列: {state_col}")
            # 将状态数据从角度转换为弧度
            states = np.array([np.array(x) for x in df[state_col]])
            # 直接对前6个维度进行弧度转换
            # states[:, :6] = np.radians(states[:, :6])
            new_df["observation.state"] = states.tolist()
        else:
            logger.warning("警告: 未找到状态数据列")
            new_df["observation.state"] = [None] * len(df)
        
        # 转换动作数据
        action_columns = [col for col in df.columns if 'action' in col.lower()]
        if action_columns:
            action_col = action_columns[0]
            logger.debug(f"使用动作列: {action_col}")
            # 将动作数据从角度转换为弧度
            actions = np.array([np.array(x) for x in df[action_col]])
            # 直接对前6个维度进行弧度转换
            # actions[:, :6] = np.radians(actions[:, :6])
            new_df["action"] = actions.tolist()
        else:
            logger.warning("警告: 未找到动作数据列")
            new_df["action"] = [None] * len(df)
        
        # 添加时间戳，根据30fps设置
        new_df["timestamp"] = np.arange(len(df)) / 30.0  # 30fps
        
        # 添加索引信息
        new_df["episode_index"] = episode_index
        new_df["frame_index"] = np.arange(len(df))
        new_df["index"] = np.arange(len(df))
        
        # 添加奖励和完成标志
        new_df["next.reward"] = (df["reward"] if "reward" in df.columns else np.zeros(len(df))).astype(np.float32)
        new_df["next.done"] = df["done"] if "done" in df.columns else np.zeros(len(df), dtype=bool)
        # 使用loc避免SettingWithCopyWarning
        new_df.loc[new_df.index[-1], "next.done"] = True  # 最后一个时间步标记为完成
        
        # 添加task_index，从数据集中读取
        new_df["task_index"] = df.get("task_index", 0) if "task_index" in df.columns else 0
        
        # 从数据集中读取列顺序，如果不存在则使用默认顺序
        column_order = df.get("column_order", [
            "observation.state",
            "action",
            "timestamp",
            "episode_index",
            "frame_index",
            "next.reward",
            "next.done",
            "index",
            "task_index"
        ]).iloc[0] if "column_order" in df.columns else [
            "observation.state",
            "action",
            "timestamp",
            "episode_index",
            "frame_index",
            "next.reward",
            "next.done",
            "index",
            "task_index"
        ]
        
        # 保存parquet文件
        output_file = data_chunk_dir / f"episode_{episode_index:06d}.parquet"
        logger.debug(f"保存parquet文件: {output_file}")
        new_df.to_parquet(output_file)
        
        logger.info(f"完成处理episode {episode_index}")
        return len(df)
        
    except Exception as e:
        logger.error(f"处理episode {episode_index}时出错: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 0

# ===================== 数据保存相关函数 =====================

def create_video(images, output_file, width=640, height=480, fps=30):
    """创建视频文件"""
    try:
        with av.open(str(output_file), mode='w') as container:
            # 使用h264编码器，它比av1更快且更广泛支持
            stream = container.add_stream('h264', rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            
            # 使用线程池并行处理图像
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                # 提交所有图像处理任务
                future_to_idx = {
                    executor.submit(process_image, img, width, height): idx 
                    for idx, img in enumerate(images)
                }
                
                # 按顺序处理结果
                processed_images = [None] * len(images)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        img = future.result()
                        if img is not None:
                            processed_images[idx] = img
                    except Exception as e:
                        logger.debug(f"处理第 {idx} 帧时出错: {e}")
                
                # 写入视频帧
                for img in processed_images:
                    if img is not None:
                        try:
                            frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                            for packet in stream.encode(frame):
                                container.mux(packet)
                        except Exception as e:
                            logger.debug(f"编码帧时出错: {e}")
            
            # 完成编码
            for packet in stream.encode():
                container.mux(packet)
                
    except Exception as e:
        logger.error(f"创建视频文件时出错: {e}")

def create_info_json(output_dir, total_episodes, total_frames, image_shapes, fps=30, input_dir=None):
    """创建info.json文件"""
    info = {
        "codebase_version": "v2.1",
        "robot_type": "xarm6",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(image_shapes),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {}
    }
    
    # 添加图像特征
    for image_col, shape in image_shapes.items():
        info["features"][f"observation.images.{image_col}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }
    
    # 添加状态特征
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [7],  # xarm6有6个关节加1个夹爪
        "names": {
            "joints": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
        }
    }
    
    # 添加动作特征
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [7],  # xarm6有6个关节加1个夹爪
        "names": {
            "joints": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
        }
    }
    
    # 添加其他特征
    for feature in ["timestamp", "episode_index", "frame_index", "next.reward", "next.done", "index", "task_index"]:
        info["features"][feature] = {
            "dtype": "float32" if feature in ["timestamp", "next.reward"] else "int64" if feature in ["episode_index", "frame_index", "index", "task_index"] else "bool",
            "shape": [1],
            "names": None
        }
    
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

def create_episodes_jsonl(output_dir, total_episodes, input_dir):
    """创建episodes.jsonl文件"""
    # 获取snapshot哈希值
    snapshot_hash = get_snapshot_hash(input_dir)
    
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for i in range(total_episodes):
            # 读取原始数据文件以获取语言指导
            snapshot_file = input_dir / "snapshots" / snapshot_hash / f"lerobot_episode_{i+1}.parquet"
            if not snapshot_file.exists():
                logger.warning(f"文件 {snapshot_file} 不存在，使用默认任务描述")
                language_instruction = "xarm6_pick_bottle_in_box"
            else:
                try:
                    # 读取原始数据文件
                    df = pd.read_parquet(snapshot_file)
                    # 从数据中获取语言指导
                    if "language_instruction" in df.columns:
                        language_instruction = df["language_instruction"].iloc[0]
                        if isinstance(language_instruction, bytes):
                            language_instruction = language_instruction.decode('utf-8')
                    else:
                        language_instruction = "xarm6_pick_bottle_in_box"
                except Exception as e:
                    logger.warning(f"读取语言指导时出错: {e}，使用默认任务描述")
                    language_instruction = "xarm6_pick_bottle_in_box"
            
            episode = {
                "episode_index": i,
                "tasks": [language_instruction],  # 使用从原始数据中读取的语言指导
                "length": 0  # 将在处理过程中更新
            }
            f.write(json.dumps(episode) + "\n")

def create_tasks_jsonl(output_dir, input_dir, total_episodes):
    """创建tasks.jsonl文件"""
    # 获取snapshot哈希值
    snapshot_hash = get_snapshot_hash(input_dir)
    
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        for i in range(total_episodes):
            # 读取每个episode以获取语言指导
            snapshot_file = input_dir / "snapshots" / snapshot_hash / f"lerobot_episode_{i+1}.parquet"
            if not snapshot_file.exists():
                logger.warning(f"文件 {snapshot_file} 不存在，使用默认任务描述")
                language_instruction = "xarm6_pick_bottle_in_box"
            else:
                try:
                    # 读取原始数据文件
                    df = pd.read_parquet(snapshot_file)
                    # 从数据中获取语言指导
                    if "language_instruction" in df.columns:
                        language_instruction = df["language_instruction"].iloc[0]
                        if isinstance(language_instruction, bytes):
                            language_instruction = language_instruction.decode('utf-8')
                    else:
                        language_instruction = "xarm6_pick_bottle_in_box"
                except Exception as e:
                    logger.warning(f"读取语言指导时出错: {e}，使用默认任务描述")
                    language_instruction = "xarm6_pick_bottle_in_box"
            
            task = {
                "task_index": i,  # 使用episode索引作为task_index
                "task": language_instruction  # 使用从原始数据中读取的语言指导
            }
            f.write(json.dumps(task) + "\n")

def create_episodes_stats_jsonl(output_dir, total_episodes, input_dir):
    """创建episodes_stats.jsonl文件"""
    logger.info("开始生成episodes_stats.jsonl...")
    
    # 获取snapshot哈希值
    snapshot_hash = get_snapshot_hash(input_dir)
    
    with open(output_dir / "meta" / "episodes_stats.jsonl", "w") as f:
        for i in tqdm(range(total_episodes), desc="生成统计信息"):
            # 读取对应的parquet文件获取数据
            parquet_file = output_dir / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
            if not parquet_file.exists():
                logger.warning(f"文件 {parquet_file} 不存在，跳过")
                continue
                
            try:
                # 读取转换后的parquet文件
                df = pd.read_parquet(parquet_file)
                
                # 读取原始数据文件以获取图像统计信息
                snapshot_file = input_dir / "snapshots" / snapshot_hash / f"lerobot_episode_{i+1}.parquet"
                if not snapshot_file.exists():
                    logger.warning(f"原始文件 {snapshot_file} 不存在，跳过")
                    continue
                    
                original_df = pd.read_parquet(snapshot_file)
                
                # 初始化stats字典
                stats = {
                    "episode_index": i,
                    "stats": {}
                }
                
                # 处理图像数据列
                image_columns = [col for col in original_df.columns if 'image' in col.lower()]
                for col in image_columns:
                    try:
                        # 只处理前100帧图像进行采样统计
                        data = original_df[col].values
                        sample_size = min(100, len(data))
                        sample_indices = np.linspace(0, len(data)-1, sample_size, dtype=int)
                        images = []
                        
                        for idx in sample_indices:
                            img_data = data[idx]
                            if isinstance(img_data, bytes):
                                nparr = np.frombuffer(img_data, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if img is not None:
                                    img = img.astype(np.float32) / 255.0
                                    images.append(img)
                            elif isinstance(img_data, np.ndarray):
                                if img_data.dtype == np.uint8:
                                    img_data = img_data.astype(np.float32) / 255.0
                                images.append(img_data)
                        
                        if images:
                            images = np.array(images)
                            # 计算每个通道的统计信息
                            min_vals = np.min(images, axis=(0, 1, 2))
                            max_vals = np.max(images, axis=(0, 1, 2))
                            mean_vals = np.mean(images, axis=(0, 1, 2))
                            std_vals = np.std(images, axis=(0, 1, 2))
                            
                            # 添加到stats字典
                            stats["stats"][f"observation.images.{col}"] = {
                                "min": [[[float(v)]] for v in min_vals],
                                "max": [[[float(v)]] for v in max_vals],
                                "mean": [[[float(v)]] for v in mean_vals],
                                "std": [[[float(v)]] for v in std_vals],
                                "count": [len(data)]  # 使用原始数据长度
                            }
                            logger.debug(f"已处理图像列 {col} 的统计信息")
                    except Exception as e:
                        logger.debug(f"计算图像列 {col} 的统计信息时出错: {e}")
                        continue
                
                # 处理其他数据列
                for col in df.columns:
                    try:
                        data = df[col].values
                        
                        if col == "next.done":
                            # 特殊处理next.done，使用布尔值
                            min_val = [bool(data.min())]
                            max_val = [bool(data.max())]
                            mean_val = [float(data.mean())]
                            std_val = [float(data.std())]
                            count = [len(data)]
                        elif col in ["episode_index", "frame_index", "index", "task_index"]:
                            # 处理元数据字段
                            min_val = [int(data.min())]
                            max_val = [int(data.max())]
                            mean_val = [float(data.mean())]
                            std_val = [float(data.std())]
                            count = [len(data)]
                        elif isinstance(data[0], (np.ndarray, list)):
                            # 对于数组类型的数据（如状态和动作）
                            data = np.array([np.array(x) for x in data])
                            min_val = data.min(axis=0).tolist()
                            max_val = data.max(axis=0).tolist()
                            mean_val = data.mean(axis=0).tolist()
                            std_val = data.std(axis=0).tolist()
                            count = [len(data)]
                        else:
                            # 对于标量类型的数据
                            min_val = [float(data.min())]
                            max_val = [float(data.max())]
                            mean_val = [float(data.mean())]
                            std_val = [float(data.std())]
                            count = [len(data)]
                        
                        # 添加到stats字典
                        stats["stats"][col] = {
                            "min": min_val,
                            "max": max_val,
                            "mean": mean_val,
                            "std": std_val,
                            "count": count
                        }
                        logger.debug(f"已处理列 {col} 的统计信息")
                    except Exception as e:
                        logger.debug(f"计算列 {col} 的统计信息时出错: {e}")
                        continue
                
                # 写入统计信息
                f.write(json.dumps(stats) + "\n")
                logger.debug(f"已处理episode {i} 的统计信息")
                
            except Exception as e:
                logger.error(f"处理episode {i} 的统计信息时出错: {e}")
                continue
    
    logger.info("episodes_stats.jsonl生成完成")



# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description="将RLDS格式的xarm6数据集转换为LerobotDataset v2.1格式")
    parser.add_argument("--input_dir", type=str, default="/home/mrblue/xarm6/xarm6_pick_bottle_in_box", 
                        help="RLDS数据集目录路径")
    parser.add_argument("--output_dir", type=str, default="/home/mrblue/xarm6_lerobot_v1", 
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
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    # 获取图像列和形状信息
    snapshot_file = input_dir / "snapshots" / get_snapshot_hash(input_dir) / "lerobot_episode_1.parquet"
    if not snapshot_file.exists():
        logger.error(f"错误: 文件 {snapshot_file} 不存在")
        return
        
    df = pd.read_parquet(snapshot_file)
    image_columns = [col for col in df.columns if 'image' in col.lower()]
    image_shapes = {}
    for col in image_columns:
        first_image = df[col].iloc[0]
        if isinstance(first_image, bytes):
            nparr = np.frombuffer(first_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                image_shapes[col] = list(img.shape)
        elif isinstance(first_image, np.ndarray):
            image_shapes[col] = list(first_image.shape)
    
    # 从原始数据中读取fps信息
    fps = 30  # 默认值
    if "fps" in df.columns:
        fps = int(df["fps"].iloc[0])
    elif "frame_rate" in df.columns:
        fps = int(df["frame_rate"].iloc[0])
    elif "sampling_rate" in df.columns:
        fps = int(df["sampling_rate"].iloc[0])
    logger.info(f"使用帧率: {fps} fps")
    
    # 自动检测episode数量
    total_episodes = get_total_episodes(input_dir)
    if total_episodes == 0:
        return
        
    # 创建元数据文件
    total_frames = 0  # 将在处理过程中计算
    create_info_json(output_dir, total_episodes, total_frames, image_shapes, fps)
    create_episodes_jsonl(output_dir, total_episodes, input_dir)
    create_tasks_jsonl(output_dir, input_dir, total_episodes)
    
    # 转换每个episode
    for i in tqdm(range(total_episodes), desc="转换episodes"):
        snapshot_file = input_dir / "snapshots" / get_snapshot_hash(input_dir) / f"lerobot_episode_{i+1}.parquet"
        if not snapshot_file.exists():
            logger.warning(f"警告: 文件 {snapshot_file} 不存在，跳过")
            continue
            
        episode_frames = convert_episode(input_dir, output_dir, i)
        if episode_frames > 0:  # 只在成功转换时更新
            total_frames += episode_frames
            
            # 更新episodes.jsonl中的length
            with open(output_dir / "meta" / "episodes.jsonl", "r") as f:
                episodes = [json.loads(line) for line in f]
            episodes[i]["length"] = episode_frames
            with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
                for episode in episodes:
                    f.write(json.dumps(episode) + "\n")
    
    # 更新info.json中的总帧数
    with open(output_dir / "meta" / "info.json", "r") as f:
        info = json.load(f)
    info["total_frames"] = total_frames
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    
    # 在所有episode处理完成后创建episodes_stats.jsonl
    create_episodes_stats_jsonl(output_dir, total_episodes, input_dir)
    
    logger.info(f"\n转换完成！数据已保存到: {output_dir}")

if __name__ == "__main__":
    main() 