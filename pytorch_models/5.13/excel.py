import pandas as pd
import os
import matplotlib.pyplot as plt

print("当前工作目录:", os.getcwd())

file_path = '/home/xzh/MYHTD/pytorch_models/5.13/5.13.txt'
excel_file_path = '/home/xzh/MYHTD/pytorch_models/5.13/data.xlsx'
# 新图像保存路径
combined_image_path = '/home/xzh/MYHTD/pytorch_models/5.13/rewards_and_collision_rates.png'
# 单独的奖励和碰撞率图像保存路径
rewards_plot_path = '/home/xzh/MYHTD/pytorch_models/5.13/rewards_plot.png'
collision_rate_plot_path = '/home/xzh/MYHTD/pytorch_models/5.13/collision_rate_plot.png'

# 初始化数据存储的列表
epochs = []
rewards = []
collision_rates = []

# 读取并处理文件
with open(file_path, 'r') as file:
    for line in file:
        # 检查行是否包含平均奖励信息
        if "Average Reward over 15 Evaluation Episodes" in line:
            parts = line.split(":")[1].split(",")  # 分割以提取数据
            reward = float(parts[0].strip())  # 提取奖励值
            collision_rate = float(parts[1].strip())  # 提取碰撞率
            epoch = len(epochs) + 1  # 确定当前的Epoch编号
            # 将数据添加到列表
            epochs.append(epoch)
            rewards.append(reward)
            collision_rates.append(collision_rate)

# 创建DataFrame
df = pd.DataFrame({
    "Epoch": epochs,
    "Average Reward": rewards,
    "Collision Rate": collision_rates
})

# 保存DataFrame到Excel文件
df.to_excel(excel_file_path, index=False)

# 创建双轴图
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Average Reward', color=color)
ax1.plot(df["Epoch"], df["Average Reward"], color=color, marker='o', linestyle='-')
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # 实例化一个第二个Y轴
color = 'tab:red'
ax2.set_ylabel('Collision Rate', color=color)
ax2.plot(df["Epoch"], df["Collision Rate"], color=color, marker='x', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # 调整布局
plt.title("Average Reward and Collision Rate Over Time")
plt.savefig(combined_image_path)
plt.show()

# 创建单独的奖励图
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Epoch"], df["Average Reward"], color='blue', marker='o', linestyle='-')
ax.set_xlabel('Epoch')
ax.set_ylabel('Average Reward')
ax.set_title('Average Reward Over Time')
plt.savefig(rewards_plot_path)
plt.show()

# 创建单独的碰撞率图
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Epoch"], df["Collision Rate"], color='red', marker='x', linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Collision Rate')
ax.set_title('Collision Rate Over Time')
plt.savefig(collision_rate_plot_path)
plt.show()

print(f'Data has been successfully saved to {excel_file_path}')
print(f'Combined plot saved to {combined_image_path}')
print(f'Reward plot saved to {rewards_plot_path}')
print(f'Collision rate plot saved to {collision_rate_plot_path}')
