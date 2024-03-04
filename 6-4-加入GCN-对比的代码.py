import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 加载数据
data_path = './data/nodes_folder/node_train-1-g6.csv'
data = pd.read_csv(data_path)

# 选择几个具有代表性的数值特征进行展示
# features_to_compare = ['selfplay_p_mcts_num', 'selfplay_step_counter', 's_runtime', 'g_runtime']
features_to_compare = ['selfplay_step_counter', 'g_runtime']

# 使用平均值填充缺失值
data_filled = data[features_to_compare].fillna(data[features_to_compare].mean())

# 应用标准化
scaler_standard = StandardScaler()
data_standardized = scaler_standard.fit_transform(data_filled)
data_standardized = pd.DataFrame(data_standardized, columns=features_to_compare)

# 应用归一化
scaler_minmax = MinMaxScaler()
data_normalized = scaler_minmax.fit_transform(data_filled)
data_normalized = pd.DataFrame(data_normalized, columns=features_to_compare)

# # 绘制归一化与标准化的箱线图对比
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# sns.boxplot(data=data_filled[features_to_compare], ax=axes[0])
# axes[0].set_title('Original Data')
# sns.boxplot(data=data_normalized, ax=axes[1], color="green")
# axes[1].set_title('Normalized Data')
# sns.boxplot(data=data_standardized, ax=axes[2], color="orange")
# axes[2].set_title('Standardized Data')

# plt.tight_layout()
# plt.show()

# # 绘制原始数据与标准化后数据的直方图对比
# fig, axes = plt.subplots(len(features_to_compare), 2, figsize=(15, 5 * len(features_to_compare)))

# for i, feature in enumerate(features_to_compare):
#     # 原始数据直方图
#     sns.histplot(data_filled[feature], kde=True, ax=axes[i, 0])
#     axes[i, 0].set_title(f'Original Distribution of {feature}')
    
#     # 标准化后数据直方图
#     sns.histplot(data_standardized[feature], kde=True, ax=axes[i, 1], color="orange")
#     axes[i, 1].set_title(f'Standardized Distribution of {feature}')

# plt.tight_layout()
# plt.show()



# 绘制归一化和标准化后的直方图对比
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, feature in enumerate(features_to_compare):
    # 原始数据直方图
    sns.histplot(data_filled[feature], kde=True, ax=axes[i, 0], color="grey")
    axes[i, 0].set_title(f'Original {feature}')
    
    # 归一化后数据直方图
    sns.histplot(data_normalized[feature], kde=True, ax=axes[i, 1], color="blue")
    axes[i, 1].set_title(f'Normalized {feature}')

    # 标准化后数据直方图
    sns.histplot(data_standardized[feature], kde=True, ax=axes[i, 2], color="orange")
    axes[i, 2].set_title(f'Standardized {feature}')

fig.savefig('./data/nodes_folder/2种预处理方法直方图对比.png')

# 绘制归一化和标准化后的箱形图对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=data_filled[features_to_compare], ax=axes[0])
axes[0].set_title('Original Data')
sns.boxplot(data=data_normalized, ax=axes[1])
axes[1].set_title('Normalized Data')
sns.boxplot(data=data_standardized, ax=axes[2])
axes[2].set_title('Standardized Data')

plt.tight_layout()

fig.savefig('./data/nodes_folder/2种预处理方法箱线图对比.png')
plt.show()