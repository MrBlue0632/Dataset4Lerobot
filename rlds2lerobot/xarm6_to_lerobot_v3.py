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
    """处理单个图像帧
    
    Args:
        image: 输入图像，可以是字节数据或numpy数组
        width: 目标宽度，默认640
        height: 目标高度，默认480
        
    Returns:
        处理后的图像，统一为(480, 640, 3)大小的numpy数组
    """
    try:
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            logger.warning(f"不支持的图像类型: {type(image)}")
            return None
            
        if img is None:
            return None
            
        # 获取原始尺寸
        h, w = img.shape[:2]
        
        # 计算目标宽高比
        target_ratio = width / height
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # 当前图像更宽，需要在宽度方向上裁剪
            new_w = int(h * target_ratio)
            start_x = (w - new_w) // 2
            img = img[:, start_x:start_x+new_w]
        elif current_ratio < target_ratio:
            # 当前图像更高，需要在高度方向上裁剪
            new_h = int(w / target_ratio)
            start_y = (h - new_h) // 2
            img = img[start_y:start_y+new_h, :]
        
        # 调整到目标大小
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        return img
        
    except Exception as e:
        logger.debug(f"处理图像时出错: {e}")
        import traceback
        logger.debug(traceback.format_exc())
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
    """创建视频文件，优化编码过程"""
    try:
        with av.open(str(output_file), mode='w') as container:
            # 使用h264编码器，配置更快的编码设置
            stream = container.add_stream('h264', rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            # 使用更快的编码预设
            stream.options = {
                'preset': 'ultrafast',
                'crf': '23',  # 平衡质量和大小
                'tune': 'zerolatency'
            }
            
            # 预处理所有图像
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                # 批量提交图像处理任务
                future_to_img = {executor.submit(process_image, img, width, height): i 
                               for i, img in enumerate(images)}
                
                # 创建结果列表
                processed_images = [None] * len(images)
                
                # 收集结果
                for future in as_completed(future_to_img):
                    idx = future_to_img[future]
                    try:
                        processed_images[idx] = future.result()
                    except Exception as e:
                        logger.debug(f"处理第 {idx} 帧时出错: {e}")
            
            # 批量编码
            frames = []
            for img in processed_images:
                if img is not None:
                    frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                    frames.append(frame)
            
            # 批量编码和写入
            packets = []
            for frame in frames:
                packets.extend(stream.encode(frame))
            
            # 写入所有数据包
            for packet in packets:
                container.mux(packet)
            
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
    
    # 添加图像特征（统一使用480x640大小）
    for image_col in image_shapes:
        info["features"][f"observation.images.{image_col}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],  # 统一的图像大小
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
                logger.warning(f"文件 {snapshot_file} 不存在，跳过该episode")
                continue
                
            try:
                # 读取原始数据文件
                df = pd.read_parquet(snapshot_file)
                logger.debug(f"读取文件 {snapshot_file} 的列: {df.columns.tolist()}")
                
                # 从数据中获取任务描述
                task_description = None
                
                # 尝试从不同的可能的列名中获取任务描述
                possible_columns = ['task_description', 'task', 'description', 'language_instruction', 'instruction', 'prompt']
                for col in possible_columns:
                    if col in df.columns and df[col].iloc[0] is not None:
                        task_description = df[col].iloc[0]
                        logger.debug(f"从列 {col} 中获取到任务描述")
                        break
                
                # 如果找到了任务描述
                if task_description is not None:
                    # 处理字节类型
                    if isinstance(task_description, bytes):
                        task_description = task_description.decode('utf-8')
                    # 确保是字符串类型并去除空白字符
                    task_description = str(task_description).strip()
                    # 如果任务描述为空字符串，视为未找到
                    if not task_description:
                        logger.warning(f"在文件 {snapshot_file} 中的任务描述为空")
                        task_description = None
                    else:
                        logger.debug(f"处理后的任务描述: {task_description}")
                else:
                    logger.warning(f"在文件 {snapshot_file} 中未找到有效的任务描述")
                    # 使用默认任务描述
                    task_description = "pick up the sprite can"
                    logger.info(f"使用默认任务描述: {task_description}")
                
                episode = {
                    "episode_index": i,
                    "tasks": [task_description],
                    "length": 0  # 将在处理过程中更新
                }
                f.write(json.dumps(episode) + "\n")
                
            except Exception as e:
                logger.error(f"处理episode {i}时出错: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

def create_tasks_jsonl(output_dir, input_dir, total_episodes):
    """创建tasks.jsonl文件"""
    # 获取snapshot哈希值
    snapshot_hash = get_snapshot_hash(input_dir)
    
    # 用于存储唯一的任务描述
    unique_tasks = {}
    
    # 首先收集所有唯一的任务描述
    for i in range(total_episodes):
        snapshot_file = input_dir / "snapshots" / snapshot_hash / f"lerobot_episode_{i+1}.parquet"
        if not snapshot_file.exists():
            logger.warning(f"文件 {snapshot_file} 不存在，跳过该episode")
            continue
            
        try:
            # 读取原始数据文件
            df = pd.read_parquet(snapshot_file)
            
            # 从数据中获取任务描述
            task_description = None
            
            # 尝试从不同的可能的列名中获取任务描述
            possible_columns = ['task_description', 'task', 'description', 'language_instruction', 'instruction', 'prompt']
            for col in possible_columns:
                if col in df.columns and df[col].iloc[0] is not None:
                    task_description = df[col].iloc[0]
                    logger.debug(f"从列 {col} 中获取到任务描述")
                    break
            
            # 如果找到了任务描述
            if task_description is not None:
                # 处理字节类型
                if isinstance(task_description, bytes):
                    task_description = task_description.decode('utf-8')
                # 确保是字符串类型并去除空白字符
                task_description = str(task_description).strip()
                # 如果任务描述为空字符串，视为未找到
                if not task_description:
                    logger.warning(f"在文件 {snapshot_file} 中的任务描述为空")
                    task_description = None
                else:
                    logger.debug(f"处理后的任务描述: {task_description}")
            else:
                logger.warning(f"在文件 {snapshot_file} 中未找到有效的任务描述")
                # 使用默认任务描述
                task_description = "pick up the sprite can"
                logger.info(f"使用默认任务描述: {task_description}")
            
            # 如果有有效的任务描述，添加到唯一任务字典中
            if task_description is not None:
                if task_description not in unique_tasks:
                    unique_tasks[task_description] = len(unique_tasks)
                    
        except Exception as e:
            logger.error(f"处理episode {i}的任务描述时出错: {e}")
            continue
    
    # 如果没有找到任何有效的任务描述
    if not unique_tasks:
        logger.error("未找到任何有效的任务描述")
        return
    
    # 写入tasks.jsonl文件
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        for task_description, task_index in unique_tasks.items():
            task = {
                "task_index": task_index,
                "task": task_description
            }
            f.write(json.dumps(task) + "\n")
            
    logger.info(f"找到 {len(unique_tasks)} 个唯一任务描述")

def process_episode_stats(args):
    """处理单个episode的统计信息
    
    Args:
        args: 包含处理所需参数的元组 (episode_index, output_dir, input_dir, snapshot_hash)
        
    Returns:
        dict: 该episode的统计信息
    """
    i, output_dir, input_dir, snapshot_hash = args
    
    try:
        # 读取对应的parquet文件获取数据
        parquet_file = output_dir / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
        if not parquet_file.exists():
            return None
            
        # 读取转换后的parquet文件
        df = pd.read_parquet(parquet_file)
        
        # 读取原始数据文件以获取图像统计信息
        snapshot_file = input_dir / "snapshots" / snapshot_hash / f"lerobot_episode_{i+1}.parquet"
        if not snapshot_file.exists():
            return None
            
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
                # 使用numpy的高效操作处理图像
                data = original_df[col].values
                sample_size = min(100, len(data))
                sample_indices = np.linspace(0, len(data)-1, sample_size, dtype=int)
                
                # 预分配内存
                images = np.zeros((sample_size, 480, 640, 3), dtype=np.float32)
                valid_count = 0
                
                for idx, sample_idx in enumerate(sample_indices):
                    img_data = data[sample_idx]
                    if isinstance(img_data, bytes):
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            # 直接调整大小并归一化
                            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                            images[valid_count] = img.astype(np.float32) / 255.0
                            valid_count += 1
                    elif isinstance(img_data, np.ndarray):
                        img = cv2.resize(img_data, (640, 480), interpolation=cv2.INTER_AREA)
                        if img_data.dtype == np.uint8:
                            images[valid_count] = img.astype(np.float32) / 255.0
                        else:
                            images[valid_count] = img
                        valid_count += 1
                
                if valid_count > 0:
                    # 只使用有效的图像进行统计
                    images = images[:valid_count]
                    
                    # 使用numpy的高效操作计算统计值
                    min_vals = np.min(images, axis=(0, 1, 2))
                    max_vals = np.max(images, axis=(0, 1, 2))
                    mean_vals = np.mean(images, axis=(0, 1, 2))
                    std_vals = np.std(images, axis=(0, 1, 2))
                    
                    stats["stats"][f"observation.images.{col}"] = {
                        "min": [[[float(v)]] for v in min_vals],
                        "max": [[[float(v)]] for v in max_vals],
                        "mean": [[[float(v)]] for v in mean_vals],
                        "std": [[[float(v)]] for v in std_vals],
                        "count": [len(data)]
                    }
            except Exception as e:
                logger.debug(f"计算图像列 {col} 的统计信息时出错: {e}")
                continue
        
        # 处理其他数据列
        for col in df.columns:
            try:
                data = df[col].values
                
                if col == "next.done":
                    # 使用numpy的布尔运算
                    stats["stats"][col] = {
                        "min": [bool(np.min(data))],
                        "max": [bool(np.max(data))],
                        "mean": [float(np.mean(data))],
                        "std": [float(np.std(data))],
                        "count": [len(data)]
                    }
                elif col in ["episode_index", "frame_index", "index", "task_index"]:
                    # 使用numpy的整数运算
                    stats["stats"][col] = {
                        "min": [int(np.min(data))],
                        "max": [int(np.max(data))],
                        "mean": [float(np.mean(data))],
                        "std": [float(np.std(data))],
                        "count": [len(data)]
                    }
                elif isinstance(data[0], (np.ndarray, list)):
                    # 对数组类型数据进行向量化处理
                    data_array = np.array([np.array(x) for x in data])
                    stats["stats"][col] = {
                        "min": np.min(data_array, axis=0).tolist(),
                        "max": np.max(data_array, axis=0).tolist(),
                        "mean": np.mean(data_array, axis=0).tolist(),
                        "std": np.std(data_array, axis=0).tolist(),
                        "count": [len(data)]
                    }
                else:
                    # 标量数据的向量化处理
                    stats["stats"][col] = {
                        "min": [float(np.min(data))],
                        "max": [float(np.max(data))],
                        "mean": [float(np.mean(data))],
                        "std": [float(np.std(data))],
                        "count": [len(data)]
                    }
            except Exception as e:
                logger.debug(f"计算列 {col} 的统计信息时出错: {e}")
                continue
        
        return stats
    except Exception as e:
        logger.error(f"处理episode {i} 的统计信息时出错: {e}")
        return None

def create_episodes_stats_jsonl(output_dir, total_episodes, input_dir):
    """创建episodes_stats.jsonl文件，使用并行处理加速统计信息生成"""
    logger.info("开始并行生成episodes_stats.jsonl...")
    
    # 获取snapshot哈希值
    snapshot_hash = get_snapshot_hash(input_dir)
    
    # 准备并行处理的参数
    args_list = [(i, output_dir, input_dir, snapshot_hash) for i in range(total_episodes)]
    
    # 使用进程池进行并行处理
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # 创建进度条
        pbar = tqdm(total=total_episodes, desc="生成统计信息")
        
        # 并行处理所有episode
        results = []
        for i, result in enumerate(pool.imap_unordered(process_episode_stats, args_list)):
            if result is not None:
                results.append(result)
            pbar.update(1)
        
        pbar.close()
    
    # 按episode_index排序结果
    results.sort(key=lambda x: x["episode_index"])
    
    # 写入结果
    with open(output_dir / "meta" / "episodes_stats.jsonl", "w") as f:
        for stats in results:
            f.write(json.dumps(stats) + "\n")
    
    logger.info(f"episodes_stats.jsonl生成完成，共处理 {len(results)} 个episodes")



# ===================== 主函数 =====================

def convert_episodes_parallel(input_dir, output_dir, total_episodes, max_workers=None):
    """并行转换多个episodes
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        total_episodes: 总episode数量
        max_workers: 最大工作进程数，默认为CPU核心数
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    logger.info(f"使用 {max_workers} 个进程进行并行处理")
    
    # 创建进度条
    pbar = tqdm(total=total_episodes, desc="转换episodes")
    
    # 使用进程池进行并行处理
    with multiprocessing.Pool(max_workers) as pool:
        # 提交所有任务
        results = []
        for i in range(total_episodes):
            result = pool.apply_async(convert_episode, 
                                    args=(input_dir, output_dir, i),
                                    callback=lambda _: pbar.update(1))
            results.append((i, result))
        
        # 收集结果
        total_frames = 0
        episodes_length = {}
        for i, result in results:
            try:
                frames = result.get()  # 等待任务完成
                if frames > 0:
                    total_frames += frames
                    episodes_length[i] = frames
            except Exception as e:
                logger.error(f"处理episode {i}时出错: {e}")
        
        pbar.close()
        
        return total_frames, episodes_length

def main():
    parser = argparse.ArgumentParser(description="将RLDS格式的xarm6数据集转换为LerobotDataset v2.1格式")
    parser.add_argument("--input_dir", type=str, default="/home/mrblue/xarm6/xarm6_pick_bottle_in_box", 
                        help="RLDS数据集目录路径")
    parser.add_argument("--output_dir", type=str, default="/home/mrblue/xarm6_lerobot_v1", 
                        help="LerobotDataset格式的输出目录")
    parser.add_argument("--debug", action="store_true", help="启用调试模式以显示详细日志")
    parser.add_argument("--width", type=int, default=640, help="处理后的图像宽度")
    parser.add_argument("--height", type=int, default=480, help="处理后的图像高度")
    parser.add_argument("--workers", type=int, default=None, help="并行处理的工作进程数，默认为CPU核心数")
    parser.add_argument("--batch_size", type=int, default=32, help="视频编码的批处理大小")
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
        # 所有图像列使用统一的目标大小
        image_shapes[col] = [args.height, args.width, 3]
    
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
    
    # 并行转换episodes
    total_frames, episodes_length = convert_episodes_parallel(
        input_dir, output_dir, total_episodes, args.workers)
    
    # 更新episodes.jsonl中的length
    with open(output_dir / "meta" / "episodes.jsonl", "r") as f:
        episodes = [json.loads(line) for line in f]
    
    for i, length in episodes_length.items():
        if i < len(episodes):
            episodes[i]["length"] = length
    
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