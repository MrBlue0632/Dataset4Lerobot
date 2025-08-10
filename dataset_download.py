import os
from huggingface_hub import snapshot_download
 
# 使用cache_dir参数，将模型/数据集保存到指定“本地路径”
snapshot_download(repo_id="数据集名称", repo_type="dataset",
                  cache_dir="./dataset",  #修改成你希望的下载路径
                  local_dir_use_symlinks=False,
                  token='对应的hf_token')
