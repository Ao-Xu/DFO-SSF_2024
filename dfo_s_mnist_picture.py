import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from dfo_s_modules import LeNet5, ScoreNet, algorithm1, algorithm2

def train_lenet5_classifier(train_dataset, num_epochs=10, model_path='mnist_lenet5.pth'):
    print(f"开始训练 LeNet-5 分类器...")
    model = LeNet5()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"  LeNet-5 训练: Epoch {epoch+1}/{num_epochs} 完成")
    torch.save(model.state_dict(), model_path)
    print(f"LeNet-5 模型已保存到 {model_path}")
    return model

def evaluate_classifier(model, test_dataset, model_name="Classifier"):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
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
        if os.path.exists(model_path):
            print(f"加载类别 {label} 的得分网络...")
            net = ScoreNet()
            net.load_state_dict(torch.load(model_path))
            score_nets[label] = net
            continue

        print(f"训练类别 {label} 的得分网络...")
        score_net = ScoreNet()
        score_net.train()
        optimizer = optim.Adam(score_net.parameters(), lr=1e-3)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        for epoch in range(num_epochs):
            for x, _ in train_loader:
                z = torch.randn_like(x) * sigma
                perturbed_x = x + z
                score_output = score_net(perturbed_x)
                loss = torch.mean(torch.sum((score_output + z / (sigma**2))**2, dim=[1,2,3]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"  Epoch {epoch+1}/{num_epochs} 完成, Loss: {loss.item():.4f}")
        torch.save(score_net.state_dict(), model_path)
        print(f"类别 {label} 的得分网络已保存。")
        score_nets[label] = score_net
    return score_nets

 
def main():
    params = {
        "beta": 0.001,     # 距离权重 
        "eta": 1.0,        # 梯度下降学习率
        "gamma": 0.01,     # 得分场强度 
        "alpha": 0.5,      # Algorithm1 的步长
        "delta": 2.0,      # DFO 扰动半径
        "kappa": 0.0       # C&W 损失函数的置信度要求
    }
    
    # --- 1. 测试实例 ---
    instance_indices = [0]
    
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 2. 数据加载 ---
    print("加载 MNIST 数据集...");
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # --- 3. 训练/加载 LeNet-5 分类器 ---
    classifier_path = 'mnist_lenet5.pth'
    if os.path.exists(classifier_path):
        classifier = LeNet5()
        classifier.load_state_dict(torch.load(classifier_path))
        print(f"加载已保存的 LeNet-5 分类器从 {classifier_path}")
    else:
        classifier = train_lenet5_classifier(train_dataset_full, model_path=classifier_path)
    
    evaluate_classifier(classifier, test_dataset, "LeNet-5")
    classifier.eval()

    # --- 4. 训练/加载所有 ScoreNets ---
    train_datasets_per_class = {i: [] for i in range(10)}
    for img, label in train_dataset_full:
        train_datasets_per_class[label].append((img, label))
    
    score_nets = train_score_networks(train_datasets_per_class)
    for net in score_nets.values(): net.eval()
    
    D_counts = {i: len(train_datasets_per_class[i]) for i in range(10)}

    for instance_index in instance_indices:
        print("-" * 50)
        print(f"正在处理索引为 {instance_index} 的实例, 参数: {params}")
        
        x0, true_label = test_dataset[instance_index]
        x0 = x0.unsqueeze(0)
        
        pred_label = classifier(x0).argmax().item()
        print(f"实例信息: 真实标签 = {true_label}, 分类器预测 = {pred_label}")

        if pred_label != true_label:
            print("警告：分类器预测错误，请选择另一个实例。")
            continue
        
        # 步骤 A: 运行 Algorithm1 寻找目标标签
        target_label = algorithm1(x0, classifier, score_nets, D_counts, alpha=params['alpha'])

        if target_label is not None:
            # 步骤 B: 运行最终版 Algorithm2 生成反事实图像
            print("运行 Algorithm 2 ：生成反事实图像...")
            counterfactual_image = algorithm2(
                x0, target_label, classifier, score_nets, D_counts,
                beta=params['beta'], eta=params['eta'], gamma=params['gamma'], delta=params['delta']
            )
            
            with torch.no_grad():
                output = classifier(counterfactual_image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, final_pred = torch.max(probabilities, 1)
                print("-" * 20)
                print(f"最终结果评估:")
                print(f"分类器对最终图像的预测: {final_pred.item()}, 置信度: {confidence.item():.2%}")
            
                # --- 可视化 ---
                plt.figure(figsize=(15, 5))

                # 子图 1: 原始图像
                plt.subplot(1, 3, 1)
                original_img = x0.squeeze().numpy() * 0.5 + 0.5
                plt.imshow(original_img, cmap='gray')
                plt.title(f'Original (Pred: {pred_label})')
                plt.axis('off')
                
                # 保存原始图像
                original_filename = f"original_index_{instance_index}.png"
                original_save_path = os.path.join(output_dir, original_filename)
                plt.imsave(original_save_path, original_img, cmap='gray')

                # 子图 2: 对抗样本
                plt.subplot(1, 3, 2)
                counterfactual_img = counterfactual_image.squeeze().detach().numpy() * 0.5 + 0.5
                plt.imshow(counterfactual_img, cmap='gray')
                plt.title(f'Counterfactual (Pred: {final_pred.item()})')
                plt.axis('off')

                # 子图 3: 扰动
                plt.subplot(1, 3, 3)
                perturbation = counterfactual_image.squeeze().detach().numpy() - x0.squeeze().numpy()
                v_min, v_max = -0.5, 0.5
                plt.imshow(perturbation, cmap='RdBu', vmin=v_min, vmax=v_max)
                plt.title(f'Perturbation (放大)')
                plt.axis('off')

                plt.tight_layout()
                
                filename = f"SUCCESS_REPLICATION_index_{instance_index}.png"
                save_path = os.path.join(output_dir, filename)
                plt.savefig(save_path)
                print(f"结果已保存到 {save_path}")
                plt.close()

        else:
            print("Algorithm 1 未能找到目标标签，处理结束。")
            
    print("-" * 50)
    print("所有选择的实例处理完毕！")

if __name__ == "__main__":
    main()
