import torch
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 模型定义 (LeNet5 和 ScoreNet) ---

class LeNet5(nn.Module):
    """
     适应 CIFAR-10 (3x32x32) 的 LeNet-5 结构实现。
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_net = nn.Sequential(
            # 输入通道从 1 (MNIST) 变为 3 (CIFAR-10)
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        # 展平尺寸从 16*5*5 (MNIST) 变为 16*6*6 (CIFAR-10)
        self.fc_net = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120), 
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_net(x)
        # 展平维度修改
        x = x.reshape(-1, 16 * 6 * 6)
        return self.fc_net(x)

# 基于全连接版本的ScoreNet
class ScoreNet(nn.Module):
    """The score network model used to guide the perturbation."""
    def __init__(self):
        super(ScoreNet, self).__init__()
        # 图像总像素点修改：28x28=784 (MNIST) 变为 3x32x32=3072 (CIFAR-10)
        input_size = 3 * 32 * 32 
        self.fc = nn.Sequential(
            nn.Linear(input_size, 2048), nn.ReLU(), # 增加中间层神经元数量以处理更大的输入
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, input_size)
        )
    def forward(self, x):
        # 展平维度修改
        x = x.view(-1, 3 * 32 * 32)
        # 还原维度修改：(-1, 1, 28, 28) 变为 (-1, 3, 32, 32)
        return self.fc(x).view(-1, 3, 32, 32) 

# --- 2. 核心算法 (迭代次数等参数调整，其余保持不变) ---

def get_score_synthesis_field(score_nets, D_counts, x, original_label):
    """Calculates a weighted average of score fields from non-original classes."""
    weights = {}
    other_labels = [i for i in range(len(D_counts)) if i != original_label]
    total_other_count = sum([D_counts.get(l, 0) for l in other_labels])
    if total_other_count + len(other_labels) == 0:
        return torch.zeros_like(x, device=device) 

    for label in other_labels:
        weights[label] = (D_counts.get(label, 0) + 1) / (total_other_count + len(other_labels))
    
    s_bar = torch.zeros_like(x, device=device) 
    for label in other_labels:
        if label in score_nets:
            s_bar += weights[label] * score_nets[label](x)
    return s_bar

def phi_cw_batch(x_batch, x0, target_label, classifier, beta=0.01, kappa=0):
    """Calculates the Carlini & Wagner (CW) loss."""
    logits = classifier(x_batch) 
    other_mask = torch.ones_like(logits, dtype=torch.bool, device=x_batch.device) 
    other_mask[:, target_label] = False
    
    max_other_logits = logits.masked_fill(~other_mask, -float('inf')).max(dim=1).values
    target_logits = logits[:, target_label]
    
    c_w_loss = torch.max(max_other_logits - target_logits, torch.tensor(-kappa, dtype=x_batch.dtype, device=x_batch.device))
    
    distance = torch.norm((x_batch - x0).view(x_batch.size(0), -1), p=2, dim=1)
    return c_w_loss + beta * distance

def dfo_gradient_vectorized(phi_func_batch, x, x0, target_label, classifier, beta, kappa, delta=0.5, N=150):
    """Estimates the gradient using a derivative-free optimization technique (Vectorized)."""
    u = torch.randn(N, *x.shape[1:], device=x.device, dtype=x.dtype)
    u_norm = torch.norm(u.view(N, -1), p=2, dim=1).view(N, 1, 1, 1)
    u = u / u_norm
    
    perturbed_batch = x + delta * u
    phi_perturbed = phi_func_batch(perturbed_batch, x0, target_label, classifier, beta=beta, kappa=kappa)
    phi_x = phi_func_batch(x, x0, target_label, classifier, beta=beta, kappa=kappa)
    
    grad_approx = torch.sum((phi_perturbed - phi_x).view(N, 1, 1, 1) * u, dim=0)
    return grad_approx / (N * delta)

def algorithm1(x0, classifier, score_nets, D_counts, alpha=0.5):
    """An untargeted attack to find the nearest vulnerable class."""
    x = x0.clone().detach()
    classifier.eval()
    original_label = classifier(x).argmax(dim=1).item()
    print(f"Algorithm 1: 原始预测标签: {original_label}")
    for i in range(1000):
        with torch.no_grad():
            s_bar = get_score_synthesis_field(score_nets, D_counts, x, original_label)
            x = x + alpha * s_bar
            x = torch.clamp(x, -1, 1)
        
        new_label = classifier(x).argmax(dim=1).item()
        if new_label != original_label:
            print(f"Algorithm 1: 在第 {i+1} 次迭代后找到目标标签: {new_label}")
            return new_label
            
    print("Algorithm 1: 未能找到目标标签。")
    return None

def algorithm2(x0, target_label, classifier, score_nets, D_counts, beta, eta, gamma, delta):
    """
    A targeted attack that finds the best adversarial example with the minimum distance (DFO-S).
    """
    x = x0.clone().detach()
    classifier.eval()
    original_label = classifier(x0).argmax(dim=1).item()
    KAPPA = 0
    
    best_adv_x = None
    best_adv_dist = float('inf')
    found_first_time = False
    
    trajectory = [x.cpu().numpy()]
    
    print(f"开始 Algorithm 2 : eta={eta}, gamma={gamma}, delta={delta}, beta={beta}")

    for i in range(2500):
        grad_phi = dfo_gradient_vectorized(
            phi_cw_batch, x, x0, target_label, classifier,
            beta=beta, kappa=KAPPA, delta=delta
        )
        
        with torch.no_grad():
            grad_phi_norm = torch.norm(grad_phi)
            exp_term = torch.exp(grad_phi_norm) - 1.0
            s_bar = get_score_synthesis_field(score_nets, D_counts, x, original_label)
            x = x - eta * grad_phi + gamma * exp_term * s_bar
            x = torch.clamp(x, -1, 1)
        
        trajectory.append(x.cpu().numpy())

        current_pred = classifier(x).argmax().item()

        if current_pred == target_label:
            if not found_first_time:
                print(f"  *** 在迭代 {i+1} 时首次找到目标标签！开始优化距离... ***")
                found_first_time = True
            
            current_dist = torch.norm((x - x0).view(-1), p=2)
            
            if current_dist < best_adv_dist:
                best_adv_dist = current_dist
                best_adv_x = x.clone().detach()
                print(f"  迭代 {i+1}: 找到更优样本。新L2距离: {current_dist:.4f}")

        if (i + 1) % 100 == 0:
            print(f"  ...进度... 迭代 {i+1}: 当前预测={current_pred}, ||∇φ̄||={grad_phi_norm:.4f}")

    print("\nAlgorithm 2 运行完成。")
    return best_adv_x, best_adv_dist.item() if best_adv_x is not None else best_adv_dist, trajectory

# --- 3. 可视化tool ---

def visualize_score_synthesis_field_pca(x0, target_label, classifier, score_nets, D_counts, num_samples=200, noise_level=0.6, output_dir="cifar10_results"):
    """
    生成一个聚合的 PCA 图，展示：样本点、原始类得分场、目标类得分场、合成得分场。
    """
    original_label = classifier(x0).argmax(dim=1).item()
    print(f"\n开始生成 Score Synthesis Field PCA 聚合图 (原标签:{original_label}, 目标标签:{target_label})...")
    
    with torch.no_grad():
        # 1. 采样点
        noise = torch.randn(num_samples, *x0.shape[1:], device=x0.device) * noise_level 
        sample_points = x0 + noise
        sample_points = torch.clamp(sample_points, -1, 1)

        # 2. 计算 Score Field
        s_bar_vectors = get_score_synthesis_field(score_nets, D_counts, sample_points, original_label)
        s_original_vectors = score_nets[original_label](sample_points) if original_label in score_nets else torch.zeros_like(s_bar_vectors)
        s_target_vectors = score_nets[target_label](sample_points) if target_label in score_nets else torch.zeros_like(s_bar_vectors)

        # 3. 准备 PCA 数据
        samples_flat = sample_points.view(num_samples, -1).cpu().numpy()
        s_bar_flat = s_bar_vectors.view(num_samples, -1).cpu().numpy()
        s_original_flat = s_original_vectors.view(num_samples, -1).cpu().numpy()
        s_target_flat = s_target_vectors.view(num_samples, -1).cpu().numpy()
        
        # 将点和得分向量全部一起拟合 PCA，确保投影空间一致
        all_data = np.vstack([samples_flat, s_bar_flat, s_original_flat, s_target_flat])
    
    # 4. PCA 降维
    pca = PCA(n_components=2)
    pca.fit(all_data)
    
    # 投影所有数据
    samples_2d = pca.transform(samples_flat)
    s_bar_2d = pca.transform(s_bar_flat) - samples_2d 
    s_original_2d = pca.transform(s_original_flat) - samples_2d 
    s_target_2d = pca.transform(s_target_flat) - samples_2d 
    
    sample_preds = classifier(sample_points).argmax(dim=1).cpu().numpy()
    
    # 5. 绘图
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(samples_2d[sample_preds == original_label, 0], samples_2d[sample_preds == original_label, 1], 
               c='blue', s=10, alpha=0.6, label=f'Predicted Class {original_label} Samples')
    ax.scatter(samples_2d[sample_preds == target_label, 0], samples_2d[sample_preds == target_label, 1], 
               c='yellow', s=10, alpha=0.6, label=f'Predicted Class {target_label} Samples')
               
    ax.quiver(samples_2d[:, 0], samples_2d[:, 1], s_bar_2d[:, 0], s_bar_2d[:, 1], 
              color='red', scale=None, scale_units='xy', alpha=0.5, 
              label='Score Synthesis Field $\\bar{s}(x)$', width=0.005)

    ax.quiver(samples_2d[:, 0], samples_2d[:, 1], s_original_2d[:, 0], s_original_2d[:, 1], 
              color='darkblue', scale=None, scale_units='xy', alpha=0.3, 
              label=f'Score Field $s_{original_label}(x)$', width=0.003)

    ax.quiver(samples_2d[:, 0], samples_2d[:, 1], s_target_2d[:, 0], s_target_2d[:, 1], 
              color='orange', scale=None, scale_units='xy', alpha=0.3, 
              label=f'Score Field $s_{target_label}(x)$', width=0.003)

    ax.set_title(f'Score Synthesis Field PCA Visualization: $x_0$ around {original_label}', fontsize=14)
    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.set_ylabel('Principal Component 2', fontsize=10)
    ax.legend()
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    pca_output_dir = os.path.join(output_dir, f'pca_viz_idx_{x0.cpu().numpy().flatten()[0]}')
    os.makedirs(pca_output_dir, exist_ok=True)
    save_path = os.path.join(pca_output_dir, f"pca_synthesis_field_idx_{original_label}_to_{target_label}.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"聚合 PCA 图已保存到: {save_path}")
    plt.close(fig)


def visualize_score_field_quiver(x0, x_cf, original_label, score_nets, D_counts, trajectory, grid_points=20, grid_range_factor=1.5, output_dir="cifar10_results"): # 🌟 添加 output_dir
    """
    Generates a quiver plot of the score synthesis field on the 2D plane,
    and plots the attack trajectory.
    """
    print(f"\n开始为标签 {original_label} 的图像生成流场箭头图 (Quiver Plot) 和轨迹...")
    
    direction_vec = x_cf - x0
    direction_vec_flat = direction_vec.flatten()
    dist = torch.norm(direction_vec_flat)
    v1_norm = direction_vec_flat / dist
    
    # 随机生成一个正交向量 v2
    rand_vec = torch.randn_like(v1_norm)
    v2_flat = rand_vec - torch.dot(rand_vec, v1_norm) * v1_norm
    v2_norm = v2_flat / torch.norm(v2_flat)

    v1 = v1_norm.view_as(x0)
    v2 = v2_norm.view_as(x0)

    x_coords = torch.linspace(-dist.item() * (grid_range_factor - 1), dist.item() * grid_range_factor, grid_points, device=x0.device)
    y_coords = torch.linspace(-dist.item(), dist.item(), grid_points, device=x0.device)
    alphas, betas = torch.meshgrid(x_coords, y_coords, indexing='xy') 

    grid_images = x0 + alphas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * v1 + \
                  betas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * v2
    grid_images_flat = grid_images.view(-1, *x0.shape[1:])

    print(f"  在 {grid_points*grid_points} 个网格点上计算得分合成场...")
    with torch.no_grad():
        s_bar_vectors = get_score_synthesis_field(score_nets, D_counts, grid_images_flat, original_label)
    
    s_bar_flat = s_bar_vectors.view(-1, s_bar_vectors.numel() // s_bar_vectors.size(0)) 
    u_components = torch.matmul(s_bar_flat, v1_norm).view(grid_points, grid_points)
    v_components = torch.matmul(s_bar_flat, v2_norm).view(grid_points, grid_points)

    fig, ax = plt.subplots(figsize=(12, 10))
    magnitudes = torch.sqrt(u_components**2 + v_components**2).cpu().numpy() 
    
    ax.quiver(alphas.cpu().numpy(), betas.cpu().numpy(), 
              u_components.cpu().numpy(), v_components.cpu().numpy(),
              mcolors.LogNorm(vmin=magnitudes.min(), vmax=magnitudes.max())(magnitudes),
              cmap='spring', 
              angles='xy', scale_units='xy', scale=None, alpha=0.5)

    trajectory_tensor = torch.tensor(np.array(trajectory), dtype=x0.dtype, device=x0.device).squeeze(1)
    num_steps = trajectory_tensor.size(0)
    
    trajectory_flat = trajectory_tensor.view(num_steps, -1)
    
    trajectory_centered = trajectory_flat - x0.view(1, -1)
    proj_x = torch.matmul(trajectory_centered, v1_norm).cpu().numpy()
    proj_y = torch.matmul(trajectory_centered, v2_norm).cpu().numpy()
    
    ax.plot(proj_x, proj_y, color='red', linestyle='-', linewidth=3, label='Solution Trajectory')
    ax.scatter(proj_x, proj_y, color='red', s=10, alpha=0.7)

    ax.plot(0, 0, 'k*', markersize=15, label='Start Point ($x_0$)') 
    ax.plot(proj_x[-1], proj_y[-1], 'kX', markersize=15, label='End Point ($x_{cf}$)')

    ax.set_title(f'Score Synthesis Field and Trajectory (Original: {original_label})', fontsize=16)
    ax.set_xlabel('Direction towards Counterfactual ($v_1$)', fontsize=12)
    ax.set_ylabel('Orthogonal Direction ($v_2$)', fontsize=12)
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    # 打印保存路径
    save_path = os.path.join(output_dir, f"score_field_quiver_trajectory_for_label_{original_label}_cf.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"流场箭头图和轨迹已保存到: {save_path}") 
    plt.close(fig)




