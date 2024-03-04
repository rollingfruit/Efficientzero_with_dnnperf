
import numpy as np
from typing import List, Dict, Tuple, Union, Any
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import os
import re


# 数据文件路径
node_folder_path = './data/nodes_folder_less'
edge_folder_path = './data/edges_folder_less'

# 指定7个特征在25个特征中的位置（减少为6种）
# 索引位置代表含义：
# 1:  selfplay_p_mcts_num
# 6:  replay_batch_size
# 11:  cpu_reanalyze_worker_count
# 14:  cpu_reanalyze_physical_count
# 16:  gpu_reanalyze_worker_count
# 19:  gpu_reanalyze_physical_count
feature_indices = [1, 6, 11, 14, 16, 19]

# 7个特征值
input_feature_values = [
    4,      # self-play中envs的并行数量
    256,    # replay 中的batch大小
    6,      # cpu的worker个数
    8,      # cpu的物理核个数
    # 6,      # gpu的worker个数
    4,      # gpu的worker个数
    2       # gpu的物理核个数
    ]


# 定义节点（运算符）和边
class Node:
    def __init__(self, node_id: int, operator_type: str, hyperparameters: Dict[str, Union[int, float]]):
        self.node_id = node_id  # 节点ID
        self.operator_type = operator_type  # 运算符类型
        self.hyperparameters = hyperparameters  

class Edge:
    def __init__(self, source_node: int, dest_node: int, weight: float = 1.0):
        self.source_node = source_node  # 源节点ID
        self.dest_node = dest_node  # 目标节点ID
        self.weight = weight  # 边的权值

# 定义DL模型为一个有向无环图（DAG）
class DLModel:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes = nodes  # 包含的节点
        self.edges = edges  # 包含的边

# ######################步骤3: 特征生成
# 定义一个安全转换函数，用于处理数据转换中的异常
def safe_convert(value, default=0):
    """ 尝试将值转换为浮点数或整数，如果失败则返回默认值 """
    try:
        # 如果值中包含小数点，则转换为浮点数，否则转换为整数
        return float(value) if '.' in value else int(value)
    except ValueError:  # 如果转换失败，捕获异常并返回默认值
        return default

# 从CSV文件生成特征
def generate_features_from_folders(node_folder_path, edge_folder_path):
    # 生成节点特征
    node_features = []
    runtimes = []
    # 读取节点文件夹中的所有CSV文件
    for node_file_name in os.listdir(node_folder_path):
        if not node_file_name.endswith('.csv'):  # 确保只处理CSV文件
            continue
        node_csv_path = os.path.join(node_folder_path, node_file_name)
        with open(node_csv_path, mode='r') as node_file:
            node_reader = csv.DictReader(node_file)
            for row in node_reader:
                # 安全转换hyperparameters中的值
                hyperparameters = {k: safe_convert(v) for k, v in row.items() if k not in ['node_type']}
                # 安全读取runtime并作为标签
                runtime_s = safe_convert(row['s_runtime']) if 's_runtime' in row else 0.0
                runtime_r = safe_convert(row['r_runtime']) if 'r_runtime' in row else 0.0
                runtime_c = safe_convert(row['c_runtime']) if 'c_runtime' in row else 0.0
                runtime_g = safe_convert(row['g_runtime']) if 'g_runtime' in row else 0.0
                runtime_t = safe_convert(row['t_runtime']) if 't_runtime' in row else 0.0
                # 使用一个简单的条件判断来找出第一个不为 0 的值
                runtime_non_zero = 0
                for runtime in [runtime_s, runtime_r, runtime_c, runtime_g, runtime_t]:
                    if runtime != 0:
                        runtime_non_zero = runtime
                        break
                runtimes.append(runtime_non_zero)

                # 生成节点特征
                node_feature = [
                    hyperparameters.get('selfplay_worker_type', 0),
                    hyperparameters.get('selfplay_p_mcts_num', 0),
                    hyperparameters.get('selfplay_step_counter', 0),
                    hyperparameters.get('selfplay_faster_counter', 0),
                    hyperparameters.get('selfplay_h_r', 0),
                    hyperparameters.get('replay_buffer_type', 0),
                    hyperparameters.get('replay_batch_size', 0),
                    hyperparameters.get('replay_total', 0),
                    hyperparameters.get('replay_buffer_length', 0),
                    hyperparameters.get('replay_h_r', 0),
                    hyperparameters.get('cpu_reanalyze_worker_type', 0),
                    hyperparameters.get('cpu_reanalyze_worker_count', 0),
                    hyperparameters.get('cpu_reanalyze_loop_count', 0),
                    hyperparameters.get('cpu_reanalyze_replay_full_count', 0),
                    hyperparameters.get('cpu_reanalyze_physical_count', 0),
                    hyperparameters.get('gpu_reanalyze_worker_type', 0),
                    hyperparameters.get('gpu_reanalyze_worker_count', 0),
                    hyperparameters.get('gpu_reanalyze_loop_count', 0),
                    hyperparameters.get('gpu_reanalyze_mcts_none_count', 0),
                    hyperparameters.get('gpu_reanalyze_physical_count', 0),
                    hyperparameters.get('train_reanalyze_worker_type', 0),
                    hyperparameters.get('train_reanalyze_is_train', 0),
                    hyperparameters.get('train_reanalyze_batch_loop_count', 0),
                    hyperparameters.get('train_reanalyze_replay_full_count', 0),
                    hyperparameters.get('train_reanalyze_buffer_size', 0),
                ]
                node_features.append(node_feature)

    # 生成边特征
    edge_features = []
    # 读取边文件夹中的所有CSV文件
    for edge_file_name in os.listdir(edge_folder_path):
        edge_csv_path = os.path.join(edge_folder_path, edge_file_name)
        with open(edge_csv_path, mode='r') as edge_file:
            edge_reader = csv.DictReader(edge_file)
            for row in edge_reader:
                # 安全转换边特征
                edge_feature = [
                    safe_convert(row['source_node'], default=-1),
                    safe_convert(row['dest_node'], default=-1),
                    safe_convert(row['weight'], default=1.0)
                ]
                edge_features.append(edge_feature)

    return np.array(node_features, dtype=np.float32), np.array(runtimes, dtype=np.float32), np.array(edge_features, dtype=np.float32)


# 执行特征生成函数
node_features, runtimes, edge_features = generate_features_from_folders(node_folder_path, edge_folder_path)

# node_features 和 runtimes 已经准备好
node_features_train, node_features_test, runtimes_train, runtimes_test = train_test_split(
    node_features, runtimes, test_size=0.2, random_state=42
)

# edge_features 已经准备好
edge_features_train, edge_features_test = train_test_split(
    edge_features, test_size=0.2, random_state=42
)

# ######################步骤4: 特征归一化
from sklearn.preprocessing import StandardScaler

# 替换原有的归一化函数
def standardize_features(features: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# 应用StandardScaler进行节点特征的标准化
scaler_node = StandardScaler().fit(node_features_train)
normalized_node_features_train = scaler_node.transform(node_features_train)
normalized_node_features_test = scaler_node.transform(node_features_test)

# 应用StandardScaler进行边特征的标准化
scaler_edge = StandardScaler().fit(edge_features_train)
normalized_edge_features_train = scaler_edge.transform(edge_features_train)
normalized_edge_features_test = scaler_edge.transform(edge_features_test)

# # 归一化节点特征
# scaler_node = MinMaxScaler().fit(node_features_train)
# normalized_node_features_train = scaler_node.transform(node_features_train)
# normalized_node_features_test = scaler_node.transform(node_features_test)

# # 归一化边特征
# scaler_edge = MinMaxScaler().fit(edge_features_train)
# normalized_edge_features_train = scaler_edge.transform(edge_features_train)
# normalized_edge_features_test = scaler_edge.transform(edge_features_test)


# ######################步骤5: Attention-based Node-Edge Encoder (ANEE) 的实现
# from torch.nn import MultiheadAttention


class ANEE(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, num_heads=4):
        super(ANEE, self).__init__()
        self.W_u = nn.Linear(node_feature_dim, hidden_dim, bias=False)
        self.W_e = nn.Linear(1, hidden_dim, bias=False)
        self.attention = nn.Parameter(torch.Tensor(1, 2 * hidden_dim))
        self.multihead_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.W_m = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.edge_feature_dim = edge_feature_dim

        nn.init.xavier_uniform_(self.attention.data)
        nn.init.xavier_uniform_(self.W_m.data)
    
    def forward(self, node_features, edge_features, edges):
        # 计算节点隐藏表示
        h_u_bar = F.leaky_relu(self.W_u(node_features))

        # 使用MultiHeadAttention处理节点特征
        h_u_bar = h_u_bar.unsqueeze(1)  # 增加一个维度以适应MultiHeadAttention
        h_u_bar, _ = self.multihead_attention(h_u_bar, h_u_bar, h_u_bar)
        h_u_bar = h_u_bar.squeeze(1)  # 去除多余的维度

        h_u = torch.zeros_like(h_u_bar)
        e_i = torch.zeros(edge_features.size(0), self.W_e.out_features, device=node_features.device)
        
        # 计算边缘影响
        for i, (src, tgt) in enumerate(edges):
            src_tgt_features = torch.cat((h_u_bar[src], h_u_bar[tgt]), dim=0)
            e_i[i] = torch.sigmoid(self.W_e(edge_features[i].unsqueeze(0)) * torch.matmul(self.attention, src_tgt_features.view(-1, 1)))

        # 聚合邻居信息
        for i, (src, tgt) in enumerate(edges):
            weighted_edges = F.softmax(e_i[i].unsqueeze(0).mm(self.W_m), dim=1)
            h_u[src] += weighted_edges.squeeze(0) * h_u_bar[tgt]

        return F.leaky_relu(h_u), e_i


# 定义模型参数
node_feature_dim = normalized_node_features_train.shape[1]
edge_true_feature_dim = normalized_edge_features_train.shape[1] - 2  # 减去edges的起点、终点列
# print(edge_true_feature_dim)
hidden_dim = 16
# 将训练集和验证集数据转换为 PyTorch 张量
node_features_tensor_train = torch.FloatTensor(normalized_node_features_train)
runtimes_tensor_train = torch.FloatTensor(runtimes_train)
node_features_tensor_test = torch.FloatTensor(normalized_node_features_test)
runtimes_tensor_test = torch.FloatTensor(runtimes_test)
edge_features_tensor_train = torch.FloatTensor(normalized_edge_features_train)
edge_features_tensor_test = torch.FloatTensor(normalized_edge_features_test)

# 或者使用NumPy
average_runtimes_test_np = np.mean(runtimes_test)
average_runtimes_tensor_test = runtimes_tensor_test.mean()
average_runtimes_500 = average_runtimes_test_np * 500
print("Average of runtimes_test: ", average_runtimes_test_np)
print("Average of runtimes_tensor_test: ", average_runtimes_tensor_test)


# Extract the source and target indices from the train and test tensors
edges_train_indices = edge_features_tensor_train[:, :2]
edges_test_indices = edge_features_tensor_test[:, :2]

# Convert to integer by rounding, assuming the original indices were integers
edges_train = [(int(src.item()), int(tgt.item())) for src, tgt in edges_train_indices]
edges_test = [(int(src.item()), int(tgt.item())) for src, tgt in edges_test_indices]

edge_features_tensor_train_third_col = edge_features_tensor_train[:, 2].unsqueeze(1)
edge_features_tensor_test_third_col = edge_features_tensor_test[:, 2].unsqueeze(1)



print(f"node_features_tensor_train的维度: {node_features_tensor_train.shape}") # torch.Size([4476, 25])

# edge_train
# 获取边的总数
num_edges = len(edges_train)
# 获取每条边的信息量（这里假设列表中至少有一条边）
info_per_edge = len(edges_train[0]) if num_edges > 0 else 0
print(f"edge_train边的总数: {num_edges}")  # 3580
print(f"edge_train每条边的信息量: {info_per_edge}") # 2

print(f"node_features_tensor_train的维度: {edge_features_tensor_train_third_col.shape}") # torch.Size([3580, 1])



# 初始化ANEE模型
anee_model = ANEE(node_feature_dim, edge_true_feature_dim, hidden_dim)
# Re-run the forward pass with the real edges for both train and test datasets
output_node_features_train, output_edge_features_train = anee_model(node_features_tensor_train, edge_features_tensor_train_third_col, edges_train)
output_node_features_test, output_edge_features_test = anee_model(node_features_tensor_test, edge_features_tensor_test_third_col, edges_test)

# ######################步骤6: 全局聚合
def global_aggregation(node_features, edge_features):
    global_node_feature = torch.mean(node_features, dim=0)
    global_edge_feature = torch.mean(edge_features, dim=0).view(1, -1)
    return global_node_feature, global_edge_feature

# 测试全局聚合函数
global_node_feature_train, global_edge_feature_train = global_aggregation(output_node_features_train, output_edge_features_train)
global_node_feature_test, global_edge_feature_test = global_aggregation(output_node_features_test, output_edge_features_test)

# ######################步骤7: 预测层
class PredictionLayer(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(PredictionLayer, self).__init__()
        self.W1 = nn.Linear(hidden_dim, output_dim)
        self.W2 = nn.Linear(hidden_dim, output_dim)
        self.bias = nn.Parameter(torch.rand(output_dim))
        
    def forward(self, global_node_feature, global_edge_feature):
        # 使用W1处理全局节点特征
        node_out = self.W1(global_node_feature)
        # print("全局节点特征处理结果: ", node_out)
        # 使用W2处理全局边特征
        edge_out = self.W2(global_edge_feature)
        # print("全局边特征处理结果: ", edge_out)
        # 将两个特征和偏置相加
        combined = node_out + edge_out + self.bias
        # print("特征合并结果: ", combined)
        # 应用ReLU激活函数
        prediction = F.relu(combined)
        # print("预测结果(ReLU激活后): ", prediction)
        # return prediction
        return prediction.view(-1, 1)  # 将输出调整为正确的形状


output_dim = 1  # 输出是一个标量值，是预测的执行时间
prediction_layer = PredictionLayer(hidden_dim, output_dim)


# 使用均方误差（MSE）作为损失函数
loss_fn = nn.MSELoss()
# 使用随机梯度下降（SGD）作为优化器


optimizer = torch.optim.AdamW(list(anee_model.parameters()) + list(prediction_layer.parameters()), lr=0.001)  # AdamW


num_epochs = 2
# progressive_loss_adjustment = lambda l, e, t: l // 10**(2 + (e // (t // 3)))

for epoch in range(num_epochs):
    optimizer.zero_grad()  # 清零梯度
    # 前向传播
    output_node_features_train, output_edge_features_train = anee_model(node_features_tensor_train, edge_features_tensor_train_third_col, edges_train)
    global_node_feature_train, global_edge_feature_train = global_aggregation(output_node_features_train, output_edge_features_train)
    prediction_train = prediction_layer(global_node_feature_train, global_edge_feature_train)
    # 计算损失
    loss_train = loss_fn(prediction_train, runtimes_tensor_train)
    # loss_train = progressive_loss_adjustment(loss_train.item(), epoch, num_epochs)

    # 反向传播
    loss_train.backward()
    optimizer.step()  # 更新参数 
    # 验证阶段
    anee_model.eval()  # 将模型设置为评估模式
    prediction_layer.eval()
    with torch.no_grad():  # 关闭梯度计算
        # 前向传播，使用验证集数据
        output_node_features_test, output_edge_features_test = anee_model(node_features_tensor_test, edge_features_tensor_test_third_col, edges_test)
        global_node_feature_test, global_edge_feature_test = global_aggregation(output_node_features_test, output_edge_features_test)
        prediction_test = prediction_layer(global_node_feature_test, global_edge_feature_test)
        # 打印global_node_feature和global_edge_feature的维度
        # print("global_node_feature_test 维度:", global_node_feature_test.shape)
        # print("global_edge_feature_test 维度:", global_edge_feature_test.shape)

        # 计算验证损失
        loss_test = loss_fn(prediction_test, runtimes_tensor_test)
        # loss_test = progressive_loss_adjustment(loss_test.item(), epoch, num_epochs)
    if epoch % 2 == 0:
            print(f"Epoch {epoch}: 训练损失 {loss_train}, 验证损失 {loss_test}")


# 使用PyTorch计算MSE和MAE
mse_test_pytorch = F.mse_loss(prediction_test, runtimes_tensor_test).item()
mae_test_pytorch = F.l1_loss(prediction_test, runtimes_tensor_test).item()
# 计算RMSE
rmse_test_pytorch = mse_test_pytorch ** 0.5

# 打印性能指标
print(f"均方误差 (MSE): {mse_test_pytorch:.6f}")
print(f"均方根误差 (RMSE): {rmse_test_pytorch:.6f}")
print(f"平均绝对误差 (MAE): {mae_test_pytorch:.6f}")


# 定义所有25个特征的默认值为0
all_features_default = [0] * 25



# 在相应的位置填充特征值
for idx, value in zip(feature_indices, input_feature_values):
    all_features_default[idx] = value

# 转换为numpy数组并进行归一化处理
input_features = np.array([all_features_default], dtype=np.float32)
normalized_input_features = scaler_node.transform(input_features)


# 转换为PyTorch张量
input_tensor = torch.FloatTensor(normalized_input_features)



def predict_runtime_type(model, node_features, edge_features, edge_indices, runtime_type):
    """ 预测特定类型的runtime """
    # 根据runtime_type调整模型输入
    # 示例：根据runtime_type调整输入特征
    # adjusted_node_features = adjust_features(node_features, runtime_type)
    model.eval()
    with torch.no_grad():
        # 预测
        output_node_features, _ = model(node_features, edge_features, edge_indices)
        global_node_feature, global_edge_feature = global_aggregation(output_node_features, torch.FloatTensor([[0]]))
        # 确保全局特征的维度正确
        global_edge_feature = global_edge_feature.view(1, -1)
        # 使用expand方法复制第二维度的值
        global_edge_feature = global_edge_feature.expand(1, 16)
        runtime = prediction_layer(global_node_feature, global_edge_feature)
    return runtime.item()

# 分别预测五种runtime
runtimes = []
for runtime_type in ['s_runtime', 'r_runtime', 'c_runtime', 'g_runtime', 't_runtime']:
    runtime = predict_runtime_type(anee_model, input_tensor, torch.FloatTensor([[0]]), [], runtime_type)
    runtimes.append(runtime)
    # print(runtime_type," is ", runtime)

# 计算单个epoch的总runtime
single_epoch_total_runtime = sum(runtimes)

# 计算100个epoch的总runtime
total_runtime_100_epochs = 100 * single_epoch_total_runtime

# print(f"100个epoch的预测总runtime: {total_runtime_100_epochs}")

def read_runtime_from_csv(csv_path):
    with open(csv_path, mode='r') as node_file:
        node_reader = csv.DictReader(node_file)
        runtimes = []
        for count, row in enumerate(node_reader):
            if count >= 500:
                break

            runtime_values = [safe_convert(row[runtime_type]) for runtime_type in ['s_runtime', 'r_runtime', 'c_runtime', 'g_runtime', 't_runtime']]
            runtime_non_zero = next((runtime for runtime in runtime_values if runtime != 0), 0.0)
            runtimes.append(runtime_non_zero)
    return np.sum(runtimes)


def process_runtimes(node_folder_path, gpu_reanalyze_worker_count=None, return_g_values=False):
    """
    处理运行时间数据。

    :param node_folder_path: 包含CSV文件的文件夹路径
    :param gpu_reanalyze_worker_count: 用于筛选文件的 GPU 工作数 (可选)
    :param return_g_values: 是否返回 g 值和运行时间的列表
    :return: 如果 return_g_values 为 False，则返回运行时间的总和；如果为 True，则返回 g 值和运行时间的列表
    """
    total_runtime = 0
    g_values = []
    r_values = []
    found_files = False

    for node_file_name in os.listdir(node_folder_path):
        match = re.search(r'g(\d+).csv', node_file_name)
        if not match:
            continue

        match_number = int(match.group(1))
        if gpu_reanalyze_worker_count is not None and match_number != gpu_reanalyze_worker_count:
            continue

        found_files = True
        node_csv_path = os.path.join(node_folder_path, node_file_name)
        runtime = read_runtime_from_csv(node_csv_path)
        total_runtime += runtime

        if return_g_values:
            g_values.append(match_number)
            r_values.append(runtime)

    if return_g_values:
        return g_values, r_values
    else:
        return total_runtime if found_files else None



def train_regression_models(g_values, r_values, threshold):
    g_values_small = [g for g, runtime in zip(g_values, r_values) if g <= threshold]
    runtimes_small = [runtime for g, runtime in zip(g_values, r_values) if g <= threshold]

    g_values_large = [g for g, runtime in zip(g_values, r_values) if g > threshold]
    runtimes_large = [runtime for g, runtime in zip(g_values, r_values) if g > threshold]

    model_small = LinearRegression().fit(np.array(g_values_small).reshape(-1, 1), runtimes_small)
    model_large = LinearRegression().fit(np.array(g_values_large).reshape(-1, 1), runtimes_large)

    return model_small, model_large

def estimate_runtime(g, model_small, model_large, threshold):
    if g <= threshold:
        return model_small.predict(np.array([[g]]))[0]
    else:
        return model_large.predict(np.array([[g]]))[0]


node_folder_path = './data/nodes_folder'
# gpu_reanalyze_worker_count = 4
gpu_reanalyze_worker_count = input_feature_values[4]

# 获取运行时间
runtimes_100 = process_runtimes(node_folder_path)
runtimes_100_can = process_runtimes(node_folder_path, gpu_reanalyze_worker_count=gpu_reanalyze_worker_count)
if runtimes_100_can:
    runtimes_100 = runtimes_100_can
# print(f"100组预测的runtime: {runtimes_100}")

# 获取 g 值和 r 值
g_values, r_values = process_runtimes(node_folder_path, return_g_values=True)

# 训练线性回归模型
fenjie = 15
total_k = 0.3
estimate_k = 1 - total_k
model_small, model_large = train_regression_models(g_values, r_values, fenjie)

# 使用模型预测
# 例如：估计 g=14 的运行时间
estimated_runtime_g14 = estimate_runtime(14, model_small, model_large, fenjie)

# 假设 total_runtime_100_epochs 是已知的
total_runtime_100_epochs = runtimes_100
final_pre_runtime = total_runtime_100_epochs * total_k + estimated_runtime_g14 * estimate_k

# print(f'Estimated runtime for g=14: {estimated_runtime_g14}')
# print(f'Real runtime for g=14: {r_values[0]}')
print(f'Actual 100_epochs runtime: {total_runtime_100_epochs}s')
print(f'Prediction 100_epochs runtime: {final_pre_runtime}s')
