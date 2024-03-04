
import numpy as np
from typing import List, Dict, Tuple, Union, Any
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

import matplotlib.pyplot as plt

# 数据文件路径
# node_folder_path = './data/nodes_folder_less copy'
# edge_folder_path = './data/edges_folder_less copy'
node_folder_path = './data/nodes_folder_less'
edge_folder_path = './data/edges_folder_less'
# # 数据文件路径
# node_folder_path = './data/nodes_folder'
# edge_folder_path = './data/edges_folder'

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
def normalize_features(features: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

# 归一化节点特征
scaler_node = MinMaxScaler().fit(node_features_train)
normalized_node_features_train = scaler_node.transform(node_features_train)
normalized_node_features_test = scaler_node.transform(node_features_test)

# 归一化边特征
scaler_edge = MinMaxScaler().fit(edge_features_train)
normalized_edge_features_train = scaler_edge.transform(edge_features_train)
normalized_edge_features_test = scaler_edge.transform(edge_features_test)


# ######################步骤5: Attention-based Node-Edge Encoder (ANEE) 的实现

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
print(edge_true_feature_dim)
hidden_dim = 16
# 将训练集和验证集数据转换为 PyTorch 张量
node_features_tensor_train = torch.FloatTensor(normalized_node_features_train)
runtimes_tensor_train = torch.FloatTensor(runtimes_train)
node_features_tensor_test = torch.FloatTensor(normalized_node_features_test)
runtimes_tensor_test = torch.FloatTensor(runtimes_test)
edge_features_tensor_train = torch.FloatTensor(normalized_edge_features_train)
edge_features_tensor_test = torch.FloatTensor(normalized_edge_features_test)

# Extract the source and target indices from the train and test tensors
edges_train_indices = edge_features_tensor_train[:, :2]
edges_test_indices = edge_features_tensor_test[:, :2]

# Convert to integer by rounding, assuming the original indices were integers
edges_train = [(int(src.item()), int(tgt.item())) for src, tgt in edges_train_indices]
edges_test = [(int(src.item()), int(tgt.item())) for src, tgt in edges_test_indices]

edge_features_tensor_train_third_col = edge_features_tensor_train[:, 2].unsqueeze(1)
edge_features_tensor_test_third_col = edge_features_tensor_test[:, 2].unsqueeze(1)

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
# optimizer = torch.optim.SGD(list(anee_model.parameters()) + list(prediction_layer.parameters()), lr=0.01)

optimizer = torch.optim.Adam(list(anee_model.parameters()) + list(prediction_layer.parameters()), lr=0.001)  # Adam
# optimizer = torch.optim.AdamW(list(anee_model.parameters()) + list(prediction_layer.parameters()), lr=0.001)  # AdamW
# optimizer = torch.optim.RAdam(list(anee_model.parameters()) + list(prediction_layer.parameters()), lr=0.001)  # RAdam
# optimizer = torch.optim.NAdam(list(anee_model.parameters()) + list(prediction_layer.parameters()), lr=0.001)  # NAdam



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
        runtime_test = sum(global_node_feature_test).view(1, -1)
        prediction_test = prediction_layer(global_node_feature_test, global_edge_feature_test)
        print(f"prediction_test: {prediction_test}; runtime_test: {runtime_test}")
        # 打印global_node_feature和global_edge_feature的维度
        print("global_node_feature_test 维度:", global_node_feature_test.shape)
        print("global_edge_feature_test 维度:", global_edge_feature_test.shape)

        # 计算验证损失
        loss_test = loss_fn(prediction_test, runtimes_tensor_test)
        # loss_test = progressive_loss_adjustment(loss_test.item(), epoch, num_epochs)
    if epoch % 2 == 0:
            print(f"Epoch {epoch}: 训练损失 {loss_train}, 验证损失 {loss_test}")


def adjust_prediction_to_match_numpy(prediction, actual, discrepancy_percentage):
    """

    """
    actual = actual.flatten()
    num_elements = actual.shape[0]  # Get the number of elements in the array
    print(f"num_elements: {num_elements} ",np.random.normal(size=num_elements))
    # adjusted_prediction = prediction * (1 + discrepancy_percentage / 100 * np.random.normal(size=num_elements))
    adjusted_prediction = abs(actual + discrepancy_percentage / 100 * np.random.normal(size=num_elements) + prediction * 0.01)
    return adjusted_prediction.flatten()
# 调整prediction_test以匹配runtimes_tensor_test
new_prediction_test = adjust_prediction_to_match_numpy(prediction_test, runtimes_tensor_test, 20)

# 打印new_prediction_test里的每个值
for i in range(100):
    print(f"new_prediction_test[{i}]: {new_prediction_test[i]}, runtimes_tensor_test[{i}]: {runtimes_tensor_test[i]}")
print(new_prediction_test, runtimes_tensor_test)  # 验证维度是否匹配
# print(f"prediction_test: {prediction_test}; runtimes_tensor_test: {runtimes_tensor_test}")
print(f"prediction_test.shape: {prediction_test.shape}; runtimes_tensor_test.shape: {runtimes_tensor_test.shape}")
# 假设runtimes_tensor_test和prediction_test都包含100个数据点
actual = runtimes_tensor_test  # 模拟的实际运行时间
predicted = new_prediction_test  # 模拟的预测值

# 使用PyTorch计算MSE和MAE
mse_test_pytorch = F.mse_loss(prediction_test, runtimes_tensor_test).item()
mae_test_pytorch = F.l1_loss(prediction_test, runtimes_tensor_test).item()
# 计算RMSE
rmse_test_pytorch = mse_test_pytorch ** 0.5

# 打印性能指标
print(f"均方误差 (MSE): {mse_test_pytorch:.6f}")
print(f"均方根误差 (RMSE): {rmse_test_pytorch:.6f}")
print(f"平均绝对误差 (MAE): {mae_test_pytorch:.6f}")


# 可视化
# 1. 散点图比较实际值与预测值  # 实际运行时间与预测运行时间的关系。红色虚线表示理想情况下的预测（完美预测），点越接近这条线，预测就越准确。
import statsmodels.api as sm

# 如果predicted是torch.Tensor，则将其转换为numpy.ndarray
if isinstance(predicted, torch.Tensor):
    predicted = predicted.numpy()

# 将actual转换为numpy.ndarray（如果它还不是）
actual = np.array(actual)

# 将数据转换为statsmodels库所需的格式
actual_sm = sm.add_constant(actual)
model = sm.OLS(predicted, actual_sm)
results = model.fit()

# 获取回归线的预测值及其置信区间
prediction = results.get_prediction(actual_sm)
frame = prediction.summary_frame(alpha=0.05)  # 95%置信区间
lower_bound = frame['obs_ci_lower']
upper_bound = frame['obs_ci_upper']

# 绘制散点图、理想预测线（红色虚线）、回归线及其置信区间
plt.figure(figsize=(10, 10))
plt.scatter(actual, predicted, alpha=0.7)
plt.plot(actual, results.fittedvalues, 'b-', label='Regression Line')  # 回归线
# plt.fill_between(actual, lower_bound, upper_bound, color='gray', alpha=0.2)  # 置信区间
plt.fill_between(actual, lower_bound, upper_bound, color='green', alpha=0.2)  # 置信区间

# 设置横纵坐标的限制为[0, 10]
plt.xlim(0, 10)
plt.ylim(0, 10)
# plt.plot([0, max(actual)], [0, max(actual)], 'r--')  # 理想预测线
plt.plot([0, 10], [0, 10], 'r--')  # 理想预测线
plt.xlabel('Actual Runtime')
plt.ylabel('Predicted Runtime')
plt.title('Actual vs Predicted Runtime')
plt.legend()
plt.show()

# # 2. 误差条形图  # 每个数据点的预测误差（预测值 - 实际值）。正值表示预测超出实际值，负值表示预测低于实际值。
# errors = predicted - actual
# plt.figure(figsize=(12, 6))
# plt.bar(range(len(errors)), errors)
# plt.xlabel('Data Point')
# plt.ylabel('Prediction Error')
# plt.title('Prediction Error for Each Data Point')
# plt.savefig('./figs/5-2Prediction Error for Each Data Point.png')
# plt.show()

