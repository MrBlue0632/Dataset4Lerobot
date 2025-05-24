# Dataset4Lerobot

两个文件夹是两种数据类型的转换，第一个针对的是强化学习的数据格式(rlds)，第二个针对的是robomind的数据格式。\

## 安装
1.如果你选择使用uv进行安装，那么你可以选择运行：
>uv sync

2.如果你选择使用pip进行安装，那么你可以使用：
>pip install -r requirements.txt

## 数据集下载
运行dataset_download.py即可，你可以选择
>uv run dataset_download.py

或者
>python dataset_download.py

**注意**：你需要先进到那个文件把参数改成你需要的

## 数据集格式说明
建议通过上面那个程序进行下载，这样你不用对那个数据集做任何处理，否则你需要手动调整成那个格式。

## 转换方法
如果你是使用uv管理环境：
>uv run Dataset4Lerobot/rlds2lerobot/xarm6_to_lerobot_v3.py --input_dir "your input dir" --ouyput_dir "your output dir"

如果你是直接用python
>python Dataset4Lerobot/rlds2lerobot/xarm6_to_lerobot_v3.py --input_dir "your input dir" --ouyput_dir "your output dir"

## 数据集上传
上传到对应的huggingface库,直接拖进去吧

## 数据集可视化
搜索lerobot/visualizedataset，输入自己库的名字
