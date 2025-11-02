import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")


# 1. 模型定义 (LeNet5 和 ScoreNet)

class LeNet5(nn.Module):
    """
     LeNet-5 结构实现。
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_net = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 16 * 5 * 5)
        return self.fc_net(x)

class ScoreNet(nn.Module):
    """
    用于估计数据分布梯度的得分网络 (基于CNN)
    整个处理过程中保持图像的2D空间结构。
    """
    def __init__(self):
        super(ScoreNet, self).__init__()
        # 'padding=1'可以保持输入和输出的 HxW 尺寸不变 (对于 kernel_size=3)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        # 最后一层使用 1x1 卷积将通道数降回 1，完成输出
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # 输入 x 的形状是 (B, 1, 28, 28)
        x = F.relu(self.conv1(x))  # -> (B, 16, 28, 28)
        x = F.relu(self.conv2(x))  # -> (B, 32, 28, 28)
        x = F.relu(self.conv3(x))  # -> (B, 16, 28, 28)
        x = self.conv_out(x)       # -> (B, 1, 28, 28)
        return x
# =========================================================
# === 2. 核心算法 (Algorithm1, phi_cw, dfo_gradient, Algorithm2)
# =========================================================

def get_score_synthesis_field(score_nets, D_counts, x, original_label):
    """
    计算得分合成场 s_bar 
    """
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

def algorithm1(x0, classifier, score_nets, D_counts, alpha=0.5):
    """
    使用得分合成场寻找目标标签 
    """
    x = x0.clone().detach() 
    classifier.eval()
    original_label = classifier(x).argmax(dim=1).item()
    print(f"Algorithm 1: 原始预测标签: {original_label}")
    for i in range(500):
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

def phi_cw(x, x0, target_label, classifier, beta=0.01, kappa=0):
    """
    C&W 损失函数，用于 DFO 的目标函数
    """
    logits = classifier(x)
    other_mask = torch.ones_like(logits, dtype=torch.bool, device=x.device) 
    other_mask[0, target_label] = False
    
    max_other_logit = logits.masked_fill(~other_mask, -float('inf')).max()
    target_logit = logits[0, target_label]
    
    c_w_loss = torch.max(max_other_logit - target_logit, torch.tensor(-kappa, dtype=x.dtype, device=x.device)) 
    
    distance = torch.norm(x.view(-1) - x0.view(-1), p=2)
    return c_w_loss + beta * distance

def dfo_gradient(phi_func, x, delta=0.5, N=150):
    """
    DFO 梯度估算 
    """
    grad_approx = torch.zeros_like(x, device=x.device)
    phi_x = phi_func(x) 
    for _ in range(N):
        u = torch.randn_like(x, device=x.device)
        u = u / torch.norm(u)
        phi_perturbed = phi_func(x + delta * u) 
        grad_approx += (phi_perturbed - phi_x) * u
    return grad_approx / (N * delta)

def algorithm2(x0, target_label, classifier, score_nets, D_counts, beta, eta, gamma, delta):
    """
    DFO-S 算法
    """
    x = x0.clone().detach() 
    classifier.eval()
    original_label = classifier(x0).argmax(dim=1).item()
    KAPPA = 0
    
    best_dist = float('inf')
    best_image = None
    
    print(f"开始 Algorithm 2: eta={eta}, gamma={gamma}, delta={delta}, beta={beta}")

    for i in range(1000):
        x.requires_grad_(True) 
        
        grad_phi = dfo_gradient(
            lambda x_val: phi_cw(x_val, x0, target_label, classifier, beta=beta, kappa=KAPPA),
            x,
            delta=delta
        )
        x.requires_grad_(False) # 关闭梯度
        
        with torch.no_grad():
            grad_phi_norm = torch.norm(grad_phi)
            
            # 2. 实现自适应开关
            exp_term = torch.exp(grad_phi_norm) - 1.0
            
            s_bar = get_score_synthesis_field(score_nets, D_counts, x, original_label)
            
            # 3. 更新公式
            x = x - eta * grad_phi + gamma * exp_term * s_bar
            x = torch.clamp(x, -1, 1)

            # 4. 跟踪最优样本
            current_pred = classifier(x).argmax().item()
            if current_pred == target_label:
                current_dist = torch.norm(x.view(-1) - x0.view(-1), p=2).item()
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_image = x.clone().detach()
                    if i % 100 != 0: 
                        print(f"  *** 在迭代 {i+1} 时首次找到目标标签！开始优化距离... ***")
                        print(f"  迭代 {i+1}: 找到更优样本。新L2距离: {best_dist:.4f}")
            
        if (i + 1) % 100 == 0:
            current_pred = classifier(x).argmax().item()
            print(f"  ...进度... 迭代 {i+1}: 当前预测={current_pred}, ||∇φ̄||={grad_phi_norm:.4f}")


    print("\nAlgorithm 2 运行完成。")
    # 返回最佳的对抗样本和其距离
    if best_image is not None:
        return best_image, best_dist
    else:
        # 如果未找到对抗样本，返回当前最后一步的 x，但距离设为 inf
        final_dist = torch.norm(x.view(-1) - x0.view(-1), p=2).item()
        return None, final_dist 











