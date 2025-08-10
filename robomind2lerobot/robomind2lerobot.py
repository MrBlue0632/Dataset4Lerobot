#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import h5py
import numpy as np
import pandas as pd
import argparse
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class RoboMind2LeRobot:
    # 配置常量
    DEFAULT_CONFIG = {
        "codebase_version": "v2.1",
        "robot_type": "robomind",
        "chunks_size": 1000,
        "fps": 15,
        "video_codec": "mp4v",
        "video_pix_fmt": "yuv420p",
        "camera_configs": {
            "rgb_static": {
                "source": "camera_front",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"]
            },
            "rgb_gripper": {
                "source": "camera_right_wrist",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"]
            }
        }
    }
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        初始化转换器
        
        Args:
            input_dir: RoboMind数据集目录
            output_dir: 输出LeRobot格式数据集目录
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建必要的目录结构
        for dir_name in ["data", "videos", "meta"]:
            (self.output_dir / dir_name).mkdir(exist_ok=True)
        
        # 初始化元数据
        self.metadata = self._init_metadata()
        
    def _init_metadata(self) -> Dict[str, Any]:
        """初始化元数据"""
        metadata = {
            "codebase_version": self.DEFAULT_CONFIG["codebase_version"],
            "robot_type": self.DEFAULT_CONFIG["robot_type"],
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 0,
            "chunks_size": self.DEFAULT_CONFIG["chunks_size"],
            "fps": self.DEFAULT_CONFIG["fps"],
            "splits": {
                "train": "0:0"  # 将在处理完成后更新
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {}
        }
        
        # 添加特征配置
        for key, config in self.DEFAULT_CONFIG["camera_configs"].items():
            metadata["features"][f"observation.images.{key}"] = {
                "dtype": "video",
                "shape": config["shape"],
                "names": config["names"],
                "video_info": {
                    "video.fps": self.DEFAULT_CONFIG["fps"],
                    "video.codec": self.DEFAULT_CONFIG["video_codec"],
                    "video.pix_fmt": self.DEFAULT_CONFIG["video_pix_fmt"],
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            }
        
        # 添加其他特征
        metadata["features"].update({
            "language_instruction": {
                "dtype": "string",
                "shape": [1],
                "names": None
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [7],
                "names": {
                    "motors": [f"motor_{i}" for i in range(7)]
                }
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {
                    "motors": [f"motor_{i}" for i in range(7)]
                }
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "next.reward": {
                "dtype": "float32",
                "shape": [1],
                "names": None
            },
            "next.done": {
                "dtype": "bool",
                "shape": [1],
                "names": None
            }
        })
        
        return metadata
        
    def _create_episodes_jsonl(self):
        """创建episodes.jsonl文件"""
        logger.info("开始生成episodes.jsonl...")
        with open(self.output_dir / "meta" / "episodes.jsonl", "w") as f:
            for i in range(self.metadata["total_episodes"]):
                # 读取对应的parquet文件获取语言指导和帧数
                chunk_index = i // self.metadata["chunks_size"]
                parquet_file = self.output_dir / "data" / f"chunk-{chunk_index:03d}" / f"episode_{i:06d}.parquet"
                language_instruction_str = "robomind_task"  # Default value
                length = 0 # Default value
                if not parquet_file.exists():
                    logger.warning(f"文件 {parquet_file} 不存在，使用默认任务描述和长度")
                else:
                    try:
                        # 读取parquet文件
                        df = pd.read_parquet(parquet_file)
                        # 从数据中获取语言指导
                        if "language_instruction" in df.columns and len(df) > 0:
                            lang_instr_val = df["language_instruction"].iloc[0]
                            # 确保语言指导是字符串类型
                            if isinstance(lang_instr_val, bytes):
                                language_instruction_str = lang_instr_val.decode('utf-8')
                            elif not isinstance(lang_instr_val, str):
                                language_instruction_str = str(lang_instr_val)
                            else:
                                language_instruction_str = lang_instr_val
                        length = len(df)
                    except Exception as e:
                        logger.warning(f"读取语言指导或帧数时出错: {e}，使用默认任务描述和长度")
                
                episode = {
                    "episode_index": i,
                    "tasks": [language_instruction_str],
                    "length": length
                }
                f.write(json.dumps(episode) + "\n")

    def _create_tasks_jsonl(self):
        """创建tasks.jsonl文件"""
        logger.info("开始生成tasks.jsonl...")
        with open(self.output_dir / "meta" / "tasks.jsonl", "w") as f:
            for i in range(self.metadata["total_episodes"]):
                # 读取对应的parquet文件获取语言指导
                chunk_index = i // self.metadata["chunks_size"]
                parquet_file = self.output_dir / "data" / f"chunk-{chunk_index:03d}" / f"episode_{i:06d}.parquet"
                language_instruction_str = "robomind_task" # Default value
                if not parquet_file.exists():
                    logger.warning(f"文件 {parquet_file} 不存在，使用默认任务描述")
                else:
                    try:
                        # 读取parquet文件
                        df = pd.read_parquet(parquet_file)
                        # 从数据中获取语言指导
                        if "language_instruction" in df.columns and len(df) > 0:
                            lang_instr_val = df["language_instruction"].iloc[0]
                            # 确保语言指导是字符串类型
                            if isinstance(lang_instr_val, bytes):
                                language_instruction_str = lang_instr_val.decode('utf-8')
                            elif not isinstance(lang_instr_val, str):
                                language_instruction_str = str(lang_instr_val)
                            else:
                                language_instruction_str = lang_instr_val
                    except Exception as e:
                        logger.warning(f"读取语言指导时出错: {e}，使用默认任务描述")
                
                task = {
                    "task_index": i,
                    "task": language_instruction_str
                }
                f.write(json.dumps(task) + "\n")

    def _create_episodes_stats_jsonl(self):
        """创建episodes_stats.jsonl文件"""
        logger.info("开始生成episodes_stats.jsonl...")
        
        with open(self.output_dir / "meta" / "episodes_stats.jsonl", "w") as f:
            for i in tqdm(range(self.metadata["total_episodes"]), desc="生成统计信息"):
                try:
                    # 读取对应的parquet文件获取数据
                    chunk_index = i // self.metadata["chunks_size"]
                    parquet_file = self.output_dir / "data" / f"chunk-{chunk_index:03d}" / f"episode_{i:06d}.parquet"
                    if not parquet_file.exists():
                        logger.warning(f"文件 {parquet_file} 不存在，跳过")
                        continue
                        
                    # 读取转换后的parquet文件
                    df = pd.read_parquet(parquet_file)
                    
                    # 初始化stats字典
                    stats = {
                        "episode_index": i,
                        "stats": {}
                    }
                    
                    # 处理图像数据列
                    image_columns = [col for col in df.columns if 'images' in col]
                    for col in image_columns:
                        try:
                            # 只处理前100帧图像进行采样统计
                            data = df[col].values
                            sample_size = min(100, len(data))
                            sample_indices = np.linspace(0, len(data)-1, sample_size, dtype=int)
                            images = []
                            
                            for idx in sample_indices:
                                img_data = data[idx]
                                if isinstance(img_data, list):
                                    img = np.array(img_data)
                                    if img.dtype == np.uint8:
                                        img = img.astype(np.float32) / 255.0
                                    images.append(img)
                            
                            if images:
                                images = np.array(images)
                                # 计算每个通道的统计信息
                                min_vals = np.min(images, axis=(0, 1, 2))
                                max_vals = np.max(images, axis=(0, 1, 2))
                                mean_vals = np.mean(images, axis=(0, 1, 2))
                                std_vals = np.std(images, axis=(0, 1, 2))
                                
                                # 添加到stats字典
                                stats["stats"][col] = {
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
                        if col not in image_columns:
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

    def convert_dataset(self):
        """转换整个数据集"""
        # 初始化总帧数计数器
        total_frames = 0
        
        # 处理训练集
        train_dir = self.input_dir / "train"
        if train_dir.exists():
            print("处理训练集...")
            total_frames = self._process_split(train_dir, "train", total_frames)
            
        # 处理验证集
        val_dir = self.input_dir / "val"
        if val_dir.exists():
            print("处理验证集...")
            total_frames = self._process_split(val_dir, "val", total_frames)
        
        # 更新chunks数量
        self.metadata["total_chunks"] = (self.metadata["total_episodes"] + self.metadata["chunks_size"] - 1) // self.metadata["chunks_size"]
        
        # 创建meta文件夹下的文件
        self._create_episodes_jsonl()
        self._create_tasks_jsonl()
        self._create_episodes_stats_jsonl()
        
        # 保存元数据
        self._save_metadata()
    
    def _process_split(self, split_dir: Path, split_name: str, start_frame: int) -> int:
        """
        处理数据集的一个划分（训练集或验证集）
        
        Args:
            split_dir: 划分目录
            split_name: 划分名称（'train' 或 'val'）
            start_frame: 起始帧索引
            
        Returns:
            更新后的总帧数
        """
        episode_count = 0
        frame_count = 0
        current_frame = start_frame
        
        # 遍历所有episode目录
        for episode_dir in sorted(split_dir.iterdir()):
            if episode_dir.is_dir():
                try:
                    frames = self._process_episode(episode_dir, split_name, self.metadata["total_episodes"] + episode_count, current_frame)
                    episode_count += 1
                    frame_count += frames
                    current_frame += frames
                except Exception as e:
                    print(f"处理episode {episode_dir} 时出错：{str(e)}")
        
        # 更新元数据
        self.metadata["total_episodes"] += episode_count
        self.metadata["total_frames"] += frame_count
        self.metadata["splits"][split_name] = f"{self.metadata['total_episodes'] - episode_count}:{self.metadata['total_episodes']}"
        
        return current_frame
    
    def _process_episode(self, episode_dir: Path, split_name: str, episode_index: int, start_frame: int) -> int:
        """
        处理单个episode的数据
        
        Args:
            episode_dir: episode目录
            split_name: 划分名称
            episode_index: episode索引
            start_frame: 起始帧索引
            
        Returns:
            处理的帧数
        """
        data_file = episode_dir / "data" / "trajectory.hdf5"
        if not data_file.exists():
            raise FileNotFoundError(f"找不到文件 {data_file}")
            
        with h5py.File(data_file, 'r') as f:
            # 读取轨迹数据
            trajectory_data = self._read_trajectory(f)
            
            # 转换为LeRobot格式
            lerobot_data = self._convert_format(trajectory_data, episode_index, start_frame)
            
            # 保存为parquet文件
            chunk_index = episode_index // self.metadata["chunks_size"]
            chunk_dir = self.output_dir / "data" / f"chunk-{chunk_index:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = chunk_dir / f"episode_{episode_index:06d}.parquet"
            self._save_parquet(lerobot_data, output_file)
            
            # 保存视频文件
            self._save_videos(trajectory_data, episode_index, chunk_index)
            
            return len(trajectory_data["master"]["joint_position_left"])
    
    def _read_trajectory(self, h5_file: h5py.File) -> Dict[str, Any]:
        """
        读取HDF5文件中的轨迹数据
        
        Args:
            h5_file: HDF5文件对象
            
        Returns:
            轨迹数据字典
        """
        data = {}
        for key in h5_file.keys():
            if isinstance(h5_file[key], h5py.Dataset):
                data[key] = h5_file[key][:]
            elif isinstance(h5_file[key], h5py.Group):
                data[key] = self._read_group(h5_file[key])
        return data
    
    def _read_group(self, group: h5py.Group) -> Dict[str, Any]:
        """
        递归读取HDF5组
        
        Args:
            group: HDF5组对象
            
        Returns:
            组数据字典
        """
        data = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                data[key] = group[key][:]
            elif isinstance(group[key], h5py.Group):
                data[key] = self._read_group(group[key])
        return data
    
    def _convert_format(self, robomind_data: Dict[str, Any], episode_index: int, start_frame: int) -> Dict[str, Any]:
        """
        将RoboMind格式转换为LeRobot格式
        
        Args:
            robomind_data: RoboMind格式的数据
            episode_index: episode索引
            start_frame: 起始帧索引
            
        Returns:
            LeRobot格式的数据
        """
        # 获取帧数
        num_frames = len(robomind_data["master"]["joint_position_left"])
        
        # 创建数据框
        data = {}
        
        # 处理图像数据 - 不存储在parquet文件中
        # 图像数据已经保存在videos目录中
        
        # 解码 language_instruction
        lang_instruction_bytes = robomind_data["language_raw"][0]
        if isinstance(lang_instruction_bytes, bytes):
            lang_instruction_str = lang_instruction_bytes.decode('utf-8')
        else:
            lang_instruction_str = str(lang_instruction_bytes)

        # 处理其他数据
        data.update({
            "language_instruction": [lang_instruction_str] * num_frames,
            "observation.state": np.column_stack([
                robomind_data["master"]["joint_position_left"],
                robomind_data["master"]["joint_position_right"]
            ]).astype(np.float32).tolist(),  # 转换为float32
            "action": np.column_stack([
                robomind_data["puppet"]["joint_position_left"],
                robomind_data["puppet"]["joint_position_right"]
            ]).astype(np.float32).tolist(),  # 转换为float32
            "timestamp": np.linspace(start_frame/self.DEFAULT_CONFIG["fps"], (start_frame + num_frames-1)/self.DEFAULT_CONFIG["fps"], num_frames, dtype=np.float32).tolist(),  # 使用连续的时间戳
            "episode_index": [episode_index] * num_frames,
            "frame_index": np.arange(num_frames).tolist(),
            "task_index": [episode_index] * num_frames,
            "index": np.arange(start_frame, start_frame + num_frames).tolist(),
            "next.reward": np.zeros(num_frames, dtype=np.float32).tolist(),  # 转换为float32
            "next.done": [False] * (num_frames - 1) + [True]  # 最后一帧为done
        })
        
        return data
    
    def _save_parquet(self, data: Dict[str, Any], output_file: Path):
        """
        保存数据为parquet格式
        
        Args:
            data: 要保存的数据
            output_file: 输出文件路径
        """
        df = pd.DataFrame(data)
        df.to_parquet(output_file)
    
    def _save_videos(self, trajectory_data: Dict[str, Any], episode_index: int, chunk_index: int):
        """
        保存视频文件
        
        Args:
            trajectory_data: 轨迹数据
            episode_index: episode索引
            chunk_index: chunk索引
        """
        for key, config in self.DEFAULT_CONFIG["camera_configs"].items():
            # 创建视频目录
            video_dir = self.output_dir / "videos" / f"chunk-{chunk_index:03d}" / f"observation.images.{key}"
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存视频
            video_path = video_dir / f"episode_{episode_index:06d}.mp4"
            frames = []
            for frame in trajectory_data["observations"]["rgb_images"][config["source"]]:
                if isinstance(frame, np.ndarray):
                    try:
                        # 尝试解码JPEG数据
                        frame_decoded = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                        if frame_decoded is not None:
                            # 调整图像大小
                            height, width = config["shape"][:2]
                            frame_resized = cv2.resize(frame_decoded, (width, height))
                            frames.append(frame_resized)
                        else:
                            print(f"警告：无法解码视频帧")
                            frames.append(np.zeros((height, width, 3), dtype=np.uint8))
                    except Exception as e:
                        print(f"警告：处理视频帧时出错 {str(e)}")
                        height, width = config["shape"][:2]
                        frames.append(np.zeros((height, width, 3), dtype=np.uint8))
                else:
                    print(f"警告：视频帧数据类型不正确")
                    height, width = config["shape"][:2]
                    frames.append(np.zeros((height, width, 3), dtype=np.uint8))
            
            if frames:
                frames = np.stack(frames)
                self._save_video(frames, video_path)
            else:
                print(f"警告：没有有效的视频帧可保存")
        
        # 更新视频计数
        self.metadata["total_videos"] += len(self.DEFAULT_CONFIG["camera_configs"])
    
    def _save_video(self, frames: np.ndarray, output_path: Path):
        """
        将图像序列保存为视频文件
        
        Args:
            frames: 图像序列
            output_path: 输出文件路径
        """
        # 获取视频参数
        height, width = frames.shape[1:3]
        fps = self.DEFAULT_CONFIG["fps"]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*self.DEFAULT_CONFIG["video_codec"])
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # 写入帧
        for frame in frames:
            # 确保图像格式正确（BGR）
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        # 释放资源
        out.release()
    
    def _save_metadata(self):
        """保存元数据"""
        metadata_file = self.output_dir / "meta" / "info.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='将RoboMind格式数据集转换为LeRobot格式')
    parser.add_argument('--input_dir', default='/home/mrblue/Projects/Robtics/Datasets/RoboMind/13_packbowl/success_episodes/', help='RoboMind数据集目录')
    parser.add_argument('--output_dir', default='/home/mrblue/Projects/Robtics/Datasets/lerobot/RobomindOutput/', help='输出LeRobot格式数据集目录')
    parser.add_argument('--debug', action='store_true', help='启用调试模式以显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    converter = RoboMind2LeRobot(args.input_dir, args.output_dir)
    converter.convert_dataset()
    print(f"转换完成！输出目录: {args.output_dir}")

if __name__ == '__main__':
    main() 