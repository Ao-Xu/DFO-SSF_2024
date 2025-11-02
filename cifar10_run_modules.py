import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import os
import sys
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 从 cifar10_modules.py 导入修改后的函数和类
try:
    from cifar10_modules import (LeNet5, ScoreNet, algorithm1, algorithm2, 
                         visualize_score_synthesis_field_pca, visualize_score_field_quiver)
except ImportError:
    print("错误: 无法导入 cifar10_modules.py 中的模块。请确保文件存在。")
    sys.exit(1)


# --- CIFAR-10 特有修改（对比MINIST） ---

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# 归一化操作
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

def unnormalize_cifar10(image_array):
    """
    将归一化的 CIFAR-10 图像 NumPy 数组反归一化回 [0, 1] 范围。
    """
    image_tensor = torch.from_numpy(image_array).to(device).float()
    
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1).to(image_tensor.device)
    
    unnormalized_tensor = image_tensor * std + mean
    
    return torch.clamp(unnormalized_tensor, 0, 1).cpu().numpy() 

# --- 通用函数 (保持不变) ---

def train_lenet5_classifier(train_dataset, num_epochs=30, model_path='cifar10_lenet5.pth'): 
    print(f"开始训练 LeNet-5 分类器...")
    model = LeNet5().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device) 
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"  LeNet-5 训练: Epoch {epoch+1}/{num_epochs} 完成, Loss: {loss.item():.4f}")
    
    model.to('cpu') 
    torch.save(model.state_dict(), model_path)
    model.to(device) 
    print(f"LeNet-5 模型已保存到 {model_path}")
    return model

def evaluate_classifier(model, test_dataset, model_name="Classifier"):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) 
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'{model_name} 在测试集上的准确率: {accuracy:.2f}%')
    return accuracy

def train_score_networks(train_datasets, sigma=0.01, num_epochs=15, output_dir='./cifar10_score_nets'): 
    os.makedirs(output_dir, exist_ok=True)
    score_nets = {}
    for label, train_dataset in train_datasets.items():
        model_path = os.path.join(output_dir, f'score_net_{label}.pth')
        
        net = ScoreNet()
        if os.path.exists(model_path):
            print(f"加载类别 {label} 的得分网络...")
            net.load_state_dict(torch.load(model_path, map_location=device)) 
            net.to(device) 
            score_nets[label] = net
            continue

        print(f"训练类别 {label} 的得分网络...")
        score_net = ScoreNet().to(device) 
        score_net.train()
        optimizer = optim.Adam(score_net.parameters(), lr=1e-3)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        for epoch in range(num_epochs):
            for x, _ in train_loader:
                x = x.to(device) 
                z = torch.randn_like(x) * sigma
                perturbed_x = x + z
                score_output = score_net(perturbed_x)
                # 切片分数匹配 Loss: 目标是让 score_output 逼近 -z / sigma**2
                loss = torch.mean(torch.sum((score_output + z / (sigma**2))**2, dim=[1,2,3]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"  Epoch {epoch+1}/{num_epochs} 完成, Loss: {loss.item():.4f}")
            
        score_net.to('cpu') 
        torch.save(score_net.state_dict(), model_path)
        score_net.to(device)
        print(f"类别 {label} 的得分网络已保存。")
        score_nets[label] = score_net
    return score_nets


def main():
    parser = argparse.ArgumentParser(description="运行基于分数的对抗攻击或可视化")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--indices', type=int, nargs='+',
                       help='需要处理的测试集样本索引列表 (例如: -i 0 15 100)')
    group.add_argument('--visualize_pca', type=int, metavar='INDEX',
                       help='为指定索引的样本生成PCA得分场聚合可视化并退出。')
    group.add_argument('--visualize_quiver', type=int, metavar='INDEX',
                       help='为指定索引的样本生成流场箭头图（需要先运行攻击）。')

    args = parser.parse_args()

    # 核心算法参数 (可调整)
    params = {
        "beta": 0.001, "eta": 1.0, "gamma": 0.1, "alpha": 0.5, "delta": 2.0
    }
    
    output_dir = "./cifar10_results" 
    os.makedirs(output_dir, exist_ok=True)
    
    print("加载 CIFAR-10 数据集...")
    # --- CIFAR-10 数据集加载 ---
    train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    classifier_path = 'cifar10_lenet5.pth' 
    classifier = LeNet5()
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location=device)) 
        print(f"加载已保存的 LeNet-5 分类器从 {classifier_path}")
    else:
        # 训练新的 CIFAR-10 分类器 (num_epochs 20-50调整，测试后增加并不能很好优化结果)
        classifier = train_lenet5_classifier(train_dataset_full, num_epochs=50, model_path=classifier_path)
    
    classifier.to(device) 
    evaluate_classifier(classifier, test_dataset, "LeNet-5 (CIFAR-10)")
    classifier.eval()
    
    # ScoreNet 训练部分 
    scorenet_output_dir = './cifar10_score_nets'
    print("为 ScoreNets 按类别准备数据集...")
    indices_per_class = {i: [] for i in range(10)}
    for i, label_val in enumerate(train_dataset_full.targets):
        indices_per_class[label_val].append(i) 

    train_datasets_per_class = {
        label: Subset(train_dataset_full, indices)
        for label, indices in indices_per_class.items()
    }
    
    score_nets = train_score_networks(train_datasets_per_class, output_dir=scorenet_output_dir) 
    for net in score_nets.values(): net.eval()
    
    D_counts = {i: len(indices) for i, indices in indices_per_class.items()}
    
    if args.indices:
        instance_indices = args.indices
    else: 
        instance_indices = [args.visualize_pca] if args.visualize_pca is not None else [args.visualize_quiver]
        
    successful_attacks = 0
    total_valid_attempts = 0

    trajectory = None
    counterfactual_image = None

    for instance_index in instance_indices:
        print("-" * 50)
        print(f"正在处理索引为 {instance_index} 的实例")
        
        x0, true_label = test_dataset[instance_index]
        x0 = x0.unsqueeze(0).to(device) 
        
        pred_label = classifier(x0).argmax().item()
        print(f"实例信息: 真实标签 = {true_label}, 分类器预测 = {pred_label}")

        if pred_label != true_label:
            print("警告：分类器对原始样本预测错误，跳过此实例。")
            continue
        
        if args.visualize_pca is not None and instance_index == args.visualize_pca:
            temp_target_label = algorithm1(x0, classifier, score_nets, D_counts, alpha=params['alpha'])
            if temp_target_label is not None:
                visualize_score_synthesis_field_pca(x0, temp_target_label, classifier, score_nets, D_counts, output_dir=output_dir)
            else:
                print("PCA可视化失败：Algorithm 1 未能找到目标标签。")
            
            if args.visualize_pca is not None: return
            continue

        total_valid_attempts += 1
        
        target_label = algorithm1(x0, classifier, score_nets, D_counts, alpha=params['alpha'])

        if target_label is not None:
            # 接收轨迹
            counterfactual_image, best_dist, trajectory = algorithm2(
                x0, target_label, classifier, score_nets, D_counts,
                beta=params['beta'], eta=params['eta'], gamma=params['gamma'], delta=params['delta']
            )
            
            if counterfactual_image is None:
                print("Algorithm 2 未能生成有效的对抗样本，本次攻击失败。")
            else:
                with torch.no_grad():
                    output = classifier(counterfactual_image) 
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, final_pred = torch.max(probabilities, 1)
                    final_pred = final_pred.item()
                    is_successful = (final_pred == target_label)
                    if is_successful:
                        successful_attacks += 1
                    
                    print("-" * 20)
                    print(f"最终结果评估:")
                    print(f"分类器对最终图像的预测: {final_pred} (目标是 {target_label})")
                    print(f"置信度: {confidence.item():.2%}")
                    print(f"本次攻击是否成功: {'是' if is_successful else '否'}")
                    print(f"最终图像与原图的L2距离: {best_dist:.4f}")
            
                    # ---  可视化部分  ---
                    fig = plt.figure(figsize=(20, 5)) 
                    
                    # 反归一化图像 (传入 NumPy 数组，函数内转换)
                    x0_unnormalized = unnormalize_cifar10(x0.squeeze().cpu().numpy())
                    cf_unnormalized = unnormalize_cifar10(counterfactual_image.squeeze().cpu().detach().numpy())
                    
                    # 1. 原始图像
                    plt.subplot(1, 4, 1)
                    plt.imshow(x0_unnormalized.transpose((1, 2, 0))) 
                    plt.title(f'Original (True: {true_label}, Pred: {pred_label})', fontsize=10)
                    plt.axis('off')
                    
                    # 2. 对抗样本
                    plt.subplot(1, 4, 2)
                    plt.imshow(cf_unnormalized.transpose((1, 2, 0))) 
                    plt.title(f'Counterfactual (Target: {target_label}, Pred: {final_pred})', fontsize=10)
                    plt.axis('off')

                    # 3. 扰动
                    plt.subplot(1, 4, 3)
                    perturbation = (counterfactual_image - x0).squeeze().cpu().detach().numpy()
                    
                    # 这里仅显示 R 通道的扰动，可调整为RBG三通道
                    perturbation_2d = perturbation[0, :, :] 
                    
                    # vmin/vmax 和MINIST暂时保持一致，可根据 CIFAR-10 的扰动范围重新调整
                    plt.imshow(perturbation_2d, cmap='RdBu_r', vmin=-0.5, vmax=0.5) 
                    plt.title(f'Perturbation R-Channel (L2: {best_dist:.4f})', fontsize=10)
                    plt.axis('off')

                    plt.subplot(1, 4, 4)
                    
                    # 基底图像（原图）
                    plt.imshow(x0_unnormalized.transpose((1, 2, 0)))

                    # 扰动数据：使用 R 通道扰动
                    perturbation_l2 = np.linalg.norm(perturbation, axis=0) # (H, W)
                    
                    abs_max = 0.5 
                    norm_perturbation_2d = np.clip(perturbation_2d / abs_max, -1, 1) # R通道归一化到 [-1, 1]
                    
                    alpha_map = np.clip(perturbation_l2 / np.max(perturbation_l2) * 1.5, 0, 1) # L2 范数归一化为 Alpha

                    # 叠加颜色图
                    overlay = plt.imshow(norm_perturbation_2d, cmap='RdBu_r', vmin=-1, vmax=1, alpha=alpha_map)
                    
                    plt.title('Original + Perturbation Overlay', fontsize=10)
                    plt.axis('off')
                    
                    cbar = fig.colorbar(overlay, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
                    cbar.set_label('Perturbation Strength (Signed)', fontsize=8)


                    plt.tight_layout()
                    save_path = os.path.join(output_dir, f"result_index_{instance_index}.png")
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"攻击结果可视化已保存到: {save_path}") 
                    plt.close(fig)

                if args.visualize_quiver is not None and instance_index == args.visualize_quiver and counterfactual_image is not None:
                    visualize_score_field_quiver(
                        x0, counterfactual_image, pred_label, 
                        score_nets, D_counts, trajectory, 
                        output_dir=output_dir
                    )
                    if args.visualize_quiver is not None: return
        else:
            print("Algorithm 1 未能找到目标标签，攻击中止。")
            
    if args.indices:
        print("=" * 50)
        print("所有选择的实例处理完毕！")
        if total_valid_attempts > 0:
            success_rate = (successful_attacks / total_valid_attempts) * 100
            print("\n--- 攻击总结 ---")
            print(f"有效尝试总数: {total_valid_attempts}")
            print(f"成功攻击次数: {successful_attacks}")
            print(f"攻击成功率: {success_rate:.2f}%")
        else:
            print("\n--- 攻击总结 ---")
            print("没有进行有效的攻击尝试。")

if __name__ == "__main__":
    main()
