import numpy as np
dataset_path = "./dataset_pod_lddmm.npy"
dataset_lddmm = np.load(dataset_path, allow_pickle=True).item() if isinstance(np.load(dataset_path, allow_pickle=True), np.lib.npyio.NpzFile) else np.load(dataset_path, allow_pickle=True)

import torch

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


plt.rcParams['font.family'] = 'Arial'
colors = ['#88c4d7', '#d0ead5', '#b5e2e5', '#9793c6', '#e6c7df', '#f79691']


import torch
import numpy as np

def load_model_and_predict(model_path, dataset_path, device='cuda'):
    """加载模型并进行预测"""
    
    print("Loading dataset...")
    # 加载数据集
    dataset = dataset_lddmm
    
    # 提取数据（根据你原代码的字段名调整）
    u_data = dataset['u_data'][:, ::2]  # 与训练时相同的采样（nskip=2）
    x_data = dataset['x_ref']  # 注意这里是'x_ref'，你给的示例中是'x_uni'，根据实际调整
    f_u = dataset["coeff_u0"]
    f_g = dataset["coeff_geo"]  # 注意这里是'coeff_geo'，你示例中是'param_data'，请根据实际数据调整
    
    # 设置参数（与训练时保持一致）
    num_umode = 75
    num_geomode = 50
    dim_br_u = [num_umode, 300, 300, 200]
    dim_br_geo = [num_geomode, 200, 200, 200]
    dim_tr = [3, 200, 200, 200]
    
    num_train = 5400
    num_test = 599
    
    # 构建时间空间网格
    print("Building time-space grid...")
    t = dataset["t"]
    t_num = t.shape[1]
    tx_ext = np.zeros((t_num, x_data.shape[0], 3))
    for i in range(t_num):
        tx_ext[i, :, 0] = t[0, i]
        tx_ext[i, :, 1:3] = x_data
    tx_ext = tx_ext[::2]  # nskip=2
    num_snap = tx_ext.shape[0]
    num_pts = x_data.shape[0]
    tx_ext = tx_ext.reshape(-1, 3)
    
    # 测试数据
    u_test = u_data[num_train:(num_train+num_test)]    
    f_u_test = f_u[num_train:(num_train+num_test)]
    f_g_test = f_g[num_train:(num_train+num_test)]
    
    # 转换为tensor
    print("Converting to tensors...")
    f_u_test_tensor = torch.tensor(f_u_test, dtype=torch.float).to(device)
    f_g_test_tensor = torch.tensor(f_g_test, dtype=torch.float).to(device)
    xt_tensor = torch.tensor(tx_ext, dtype=torch.float).to(device)
    
    # 定义模型类（需要与训练时的模型结构一致）
    class opnn(torch.nn.Module):
        def __init__(self, branch1_dim, branch2_dim, trunk_dim):
            super(opnn, self).__init__()
            
            # Build branch net for branch1
            modules = []
            in_channels = branch1_dim[0]
            for h_dim in branch1_dim[1:]:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(in_channels, h_dim),
                    torch.nn.Tanh()
                ))
                in_channels = h_dim
            self._branch1 = torch.nn.Sequential(*modules)

            # Build branch net for branch2
            modules = []
            in_channels = branch2_dim[0]
            for h_dim in branch2_dim[1:]:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(in_channels, h_dim),
                    torch.nn.Tanh()
                ))
                in_channels = h_dim
            self._branch2 = torch.nn.Sequential(*modules)

            # Build trunk net
            modules = []
            in_channels = trunk_dim[0]
            for h_dim in trunk_dim[1:]:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(in_channels, h_dim),
                    torch.nn.Tanh()
                ))
                in_channels = h_dim
            self._trunk = torch.nn.Sequential(*modules)

        def forward(self, f, f_bc, x):
            y_br1 = self._branch1(f)
            y_br2 = self._branch2(f_bc)
            y_br = y_br1 * y_br2

            y_tr = self._trunk(x)
            y_out = torch.einsum("ij,kj->ik", y_br, y_tr)
            return y_out
    
    # 加载模型
    print("Loading model...")
    model = opnn(dim_br_u, dim_br_geo, dim_tr).to(device)
    model = model.float()
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 进行预测
    print("Making predictions...")
    with torch.no_grad():
        u_test_pred = model.forward(f_u_test_tensor, f_g_test_tensor, xt_tensor)
        u_test_pred = u_test_pred.reshape(-1, num_snap, num_pts)
        u_pred = u_test_pred.detach().cpu().numpy()
    
    print(f"Prediction completed! Shape: {u_pred.shape}")
    return u_test, u_pred


def visualize_pca_single_model(u_true, u_pred, save_path="./pca_results", model_name="Model"):
    """
    对单个模型进行PCA可视化，仅绘制真实值与预测值在PCA空间中的重叠图。
    """
    
    import os
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    
    # 数据预处理：展平时空维度
    num_samples = u_true.shape[0]
    u_true_flat = u_true.reshape(num_samples, -1)
    u_pred_flat = u_pred.reshape(num_samples, -1)
    
    # 合并数据进行PCA拟合
    all_data_flat = np.vstack([u_true_flat, u_pred_flat])
    
    # 标准化
    scaler = StandardScaler()
    all_data_scaled = scaler.fit_transform(all_data_flat)
    
    # PCA降维到2D
    pca = PCA(n_components=2)
    all_data_pca = pca.fit_transform(all_data_scaled)
    
    # 分离PCA结果
    u_true_pca = all_data_pca[:num_samples]
    u_pred_pca = all_data_pca[num_samples:]
    
    # 绘制重叠图
    plt.figure(figsize=(10, 8))
    plt.scatter(u_true_pca[:, 0], u_true_pca[:, 1], 
                c=colors[0], alpha=0.5, s=100, label='Ground Truth', marker='o')
    plt.scatter(u_pred_pca[:, 0], u_pred_pca[:, 1], 
                c=colors[5], alpha=0.5, s=100, label='Predicted', marker='o')
    
    plt.xlabel(f'PC1', fontsize=17) # ({pca.explained_variance_ratio_[0]:.1%} variance)
    plt.ylabel(f'PC2', fontsize=17) # ({pca.explained_variance_ratio_[1]:.1%} variance)

    plt.ylim(-230, 300)

    plt.title('Ground Truth vs Predicted', fontsize=20, fontweight='bold', pad=15)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.5, lw=2, ls='--')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{model_name}/pca_comparison.png', dpi=600)
    plt.savefig(f'{save_path}/{model_name}/pca_comparison.svg', dpi=600)
    plt.savefig(f'{save_path}/{model_name}/pca_comparison.pdf', dpi=600)

    # plt.show()

def main():
    # 设置路径
    dataset_path = ""
    model_path = './pt/old_3.pt'
    save_path = "./pca_visualization"
    
    print("Starting PCA visualization...")
    print(f"Dataset path: {dataset_path}")
    print(f"Model path: {model_path}")
    print(f"Save path: {save_path}")
    
    u_true, u_pred = load_model_and_predict(model_path, dataset_path, device='cpu')

    

    # 进行PCA可视化
    print("Starting PCA visualization...")
    results = visualize_pca_single_model(
        u_true=u_true,
        u_pred=u_pred,
        save_path=save_path,
        model_name="old_3_Model"
    )
    
    print("PCA visualization completed!")
    print(f"Results saved to: {save_path}")

if __name__ == "__main__":
    main()