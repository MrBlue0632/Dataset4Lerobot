# RoboMind 到 LeRobot 数据集转换工具

这个工具用于将 RoboMind 格式的数据集转换为 LeRobot 格式。

## 安装

1. 确保您已安装 Python 3.6 或更高版本
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python robomind2lerobot.py <input_dir> <output_dir>
```

参数说明：
- `input_dir`: RoboMind 格式数据集的目录路径
- `output_dir`: 转换后的 LeRobot 格式数据集输出目录路径

## 示例

```bash
python robomind2lerobot.py ./robomind_dataset ./lerobot_dataset
```

## 注意事项

- 请确保输入目录中包含有效的 RoboMind 格式数据
- 输出目录如果不存在会自动创建
- 转换过程会保持原始目录结构 