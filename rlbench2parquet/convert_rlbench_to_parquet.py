import os
import pickle
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import argparse

# 禁用 Arrow 扩展类型以防保存失败
os.environ["PANDAS_USE_PYARROW_EXTENSION_ARRAY"] = "0"
os.environ["PANDAS_ARROW_EXTENSION_TYPES"] = "0"

def convert_episode_to_parquet(episode_dir, parquet_output_path, language_prompt, image_dirs):
    #actions_path = os.path.join(episode_dir, "actions.npy")
    obs_path = os.path.join(episode_dir, "low_dim_obs.pkl")
    #actions = np.load(actions_path)

    with open(obs_path, "rb") as f:
        obs_all = pickle.load(f)
        joints = [obs.joint_positions for obs in obs_all]

    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", token="your_token")
    _ = tokenizer(language_prompt, padding="max_length", truncation=True, max_length=20)

    image_file_lists = []
    for image_dir in image_dirs:
        full_dir = os.path.join(episode_dir, image_dir)
        files = sorted(os.listdir(full_dir), key=lambda x: int(os.path.splitext(x)[0]))
        image_file_lists.append((full_dir, files))

    num_steps = len(joints)
    rows = []

    for t in range(num_steps):
        row = {}
        for i, (img_dir, file_list) in enumerate(image_file_lists):
            img_path = os.path.join(img_dir, file_list[t])
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            row[f"image_{i+1}"] = img_bytes

        obs = obs_all[t]
        state_vec = obs.joint_positions.tolist() + [obs.gripper_open]

        if (t+1)!= num_steps:
            obs_future = obs_all[t+1]
            action_vec = obs_future.joint_positions.tolist() + [obs_future.gripper_open]
        else:
            action_vec = obs.joint_positions.tolist() + [obs.gripper_open]

        row.update({
            "state": state_vec,
            "action": action_vec,
            "is_first": (t == 0),
            "is_last": (t == num_steps - 1),
            "is_terminal": (t == num_steps - 1),
            "prompt": language_prompt
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_output_path, index=False)
    print(f"[✓] Saved: {parquet_output_path}")

def batch_convert_episodes(root_episodes_dir, output_dir, language_prompt):
    image_dirs = ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb", "overhead_rgb"]
    #image_dirs = ["front_rgb", "wrist_rgb", "overhead_rgb"]

    os.makedirs(output_dir, exist_ok=True)

    for entry in os.listdir(root_episodes_dir):
        episode_path = os.path.join(root_episodes_dir, entry)
        if os.path.isdir(episode_path) and entry.startswith("episode"):
            episode_number = ''.join(filter(str.isdigit, entry))
            output_parquet_path = os.path.join(output_dir, f"lerobot_episode_{episode_number}.parquet")
            convert_episode_to_parquet(episode_path, output_parquet_path, language_prompt, image_dirs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="处理Parquet数据的脚本")
    parser.add_argument("--input_dir", type=str,required = True)
    parser.add_argument("--out_dir", type=str,required = True)
    parser.add_argument("--prompt", type=str,required = True)
    args = parser.parse_args()
    #root_dir = "/home/bozhao4060/code/ljt/Datasets/close_box/variation0/episodes"
    #out_dir = "/home/bozhao4060/code/ljt/Datasets/output"
    #prompt = "pick the fanta and put it in the basket"
    batch_convert_episodes(args.input_dir,args.out_dir,args.prompt)
