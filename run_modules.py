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

try:
    from modules import (LeNet5, ScoreNet, algorithm1, algorithm2, 
                         visualize_score_synthesis_field_pca, visualize_score_field_quiver)
except ImportError:
    print("错误: 无法导入 modules.py 中的模块。请确保 modules.py 文件存在并包含所有必要的函数和类。")
    sys.exit(1)


def train_lenet5_classifier(train_dataset, num_epochs=10, model_path='mnist_lenet5.pth'):
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

def train_score_networks(train_datasets, sigma=0.01, num_epochs=10, output_dir='./score_nets'):
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

    # 核心算法参数
    params = {
        "beta": 0.001, "eta": 1.0, "gamma": 0.01, "alpha": 0.5, "delta": 2.0
    }
    
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("加载 MNIST 数据集...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    classifier_path = 'mnist_lenet5.pth'
    classifier = LeNet5()
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location=device)) 
        print(f"加载已保存的 LeNet-5 分类器从 {classifier_path}")
    else:
        classifier = train_lenet5_classifier(train_dataset_full, num_epochs=10, model_path=classifier_path)
    
    classifier.to(device) 
    evaluate_classifier(classifier, test_dataset, "LeNet-5")
    classifier.eval()
    
    print("为 ScoreNets 按类别准备数据集...")
    indices_per_class = {i: [] for i in range(10)}
    for i, label_val in enumerate(train_dataset_full.targets):
        indices_per_class[label_val.item()].append(i)

    train_datasets_per_class = {
        label: Subset(train_dataset_full, indices)
        for label, indices in indices_per_class.items()
    }
    
    score_nets = train_score_networks(train_datasets_per_class) 
    for net in score_nets.values(): net.eval()
    
    D_counts = {i: len(indices) for i, indices in indices_per_class.items()}
    
    if args.indices:
        instance_indices = args.indices
    else: 
        instance_indices = [args.visualize_pca] if args.visualize_pca is not None else [args.visualize_quiver]
        
    successful_attacks = 0
    total_valid_attempts = 0

    # 预初始化轨迹变量，以防未运行 Algorithm 2
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
            # Algorithm 1 必须运行来获取目标标签，才能进行 PCA 可视化
            temp_target_label = algorithm1(x0, classifier, score_nets, D_counts, alpha=params['alpha'])
            if temp_target_label is not None:
                pca_output_dir = os.path.join(output_dir, f'pca_viz_idx_{instance_index}')
                visualize_score_synthesis_field_pca(x0, temp_target_label, classifier, score_nets, D_counts, output_dir=pca_output_dir)
            else:
                print("PCA可视化失败：Algorithm 1 未能找到目标标签。")
            
            if args.visualize_pca is not None: return
            continue

        total_valid_attempts += 1
        
        target_label = algorithm1(x0, classifier, score_nets, D_counts, alpha=params['alpha'])

        if target_label is not None:
            #  接收轨迹
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
            
                    # ---  可视化部分---
                    fig = plt.figure(figsize=(20, 5)) 
                    
                    # 1. 原始图像
                    plt.subplot(1, 4, 1)
                    plt.imshow(x0.squeeze().cpu().numpy() * 0.5 + 0.5, cmap='gray') 
                    plt.title(f'Original (True: {true_label}, Pred: {pred_label})', fontsize=10)
                    plt.axis('off')
                    
                    # 2. 对抗样本
                    plt.subplot(1, 4, 2)
                    plt.imshow(counterfactual_image.squeeze().cpu().detach().numpy() * 0.5 + 0.5, cmap='gray') 
                    plt.title(f'Counterfactual (Target: {target_label}, Pred: {final_pred})', fontsize=10)
                    plt.axis('off')

                    # 3. 扰动 
                    plt.subplot(1, 4, 3)
                    perturbation = (counterfactual_image - x0).squeeze().cpu().detach().numpy() 
                    plt.imshow(perturbation, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
                    plt.title(f'Perturbation (L2: {best_dist:.4f})', fontsize=10)
                    plt.axis('off')

                    plt.subplot(1, 4, 4)
                    
                    # 基底图像
                    plt.imshow(x0.squeeze().cpu().numpy() * 0.5 + 0.5, cmap='gray') 

                    # 扰动数据（使用 RdBu_r 颜色映射来表示符号和幅度，并用 alpha 叠加）
                    abs_max = 0.5 
                    norm_perturbation = np.clip(perturbation / abs_max, -1, 1)
                    
                    # 创建一个 alpha 映射：扰动强度越大，透明度越高
                    alpha_map = np.abs(norm_perturbation)
                    
                    # 叠加颜色图
                    overlay = plt.imshow(norm_perturbation, cmap='RdBu_r', vmin=-1, vmax=1, alpha=alpha_map)
                    
                    plt.title('Original + Perturbation Overlay', fontsize=10)
                    plt.axis('off')
                    
                    # 添加颜色条（Colorbar）
                    cbar = fig.colorbar(overlay, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
                    cbar.set_label('Perturbation Strength (Signed)', fontsize=8)


                    plt.tight_layout()
                    save_path = os.path.join(output_dir, f"result_index_{instance_index}.png")
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"结果已保存到 {save_path}")
                    plt.close(fig)

                # 流场箭头图可视化
                if args.visualize_quiver is not None and instance_index == args.visualize_quiver and counterfactual_image is not None:
                    visualize_score_field_quiver(
                        x0, counterfactual_image, pred_label, 
                        score_nets, D_counts, trajectory
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


