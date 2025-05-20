# 代码转换文件说明文档（边用边写）
## 数据集格式说明（别看，没用）
1.RLDS全称Reinforcement Learning Datasets，也就是强化学习的常用数据集

2.具体格式说明文档
> Dataset\
> ├── Episode 1\
> │   ├── Step 1\
> │   ├── Step 2\
> │   └── ...
> │\
> ├── Episode 2\
> │   ├── Step 1\
> │   ├── Step 2\
> │   └── ...\
> └── ...

如上图，基于episode进行存储。其中，给一个step包含如下几个字段（可能多可能少）

>'observation': ...,       环境观测\
'action': ...,            采取的动作\
'reward': ...,           获得的奖励\
'discount': ...,          折扣因子\
'is_first': ...,         是否为episode第一步\
'is_last': ...,           是否为episode最后一步\
'is_terminal': ...        是否为终止状态\

好吧，上面提到的东西是网上查到的标准格式，当然实际上我们拿到的不是这样的。
我们的数据实际上是：
>.cache : 这里面放的是huggingface相关的文件，具体是什么我也不知道\
> refs : 这是一个文件夹，里面有一个名叫main的文件，且没有后缀。这里面我们可以写一个字符串,比如abc\
>snapshots : 文件夹，里面再嵌套一个文件夹，这个嵌套文件夹的名字就是我们main里面的字符串，这里就是abc。
> 我们把所有的数据，也就是一大堆xx.parquet放在里面。

## 运行方法
进入对应文件夹，然后输入下面的指令
>python xarm6_to_lerobot_v2.py --input_dir /path/to/xarm6_pick_bottle_in_box --output_dir /path/to/output

## 转换结果
如果一切正常，你会在你的output_dir里面得到三个文件夹
>data : 下面有一个chunk-000的文件夹，里面放了很多xx.parquet的文件\
>meta : 这里面会有四个配置文件，都是jsonl\
>videos ：这里面也有一个chunk-000,里面放了两个文件夹的video（虽然标注的是image）。根据观察，1里面的应该是全局视角，2里面的应该是
>局部视角

## 可视化方法
在huggingface上面搭建一个库，把自己的数据集放进去。搜索lerobot/visualizedataset，然后进去输入自己的库的名字，就可以线上可视化了
（我感觉这个要简单一些）