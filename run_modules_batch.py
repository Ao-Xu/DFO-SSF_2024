import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from modules import LeNet5, ScoreNet, algorithm1, algorithm2

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
        "beta": 0.001,
        "eta": 1.0,
        "gamma": 0.01,
        "alpha": 0.5,
        "delta": 2.0,
        "kappa": 0.0
    }
    
    num_instances_to_test = 10
    instance_indices = range(num_instances_to_test)
    
    output_dir = "./results_batch"
    os.makedirs(output_dir, exist_ok=True)
    
    print("加载 MNIST 数据集...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    classifier_path = 'mnist_lenet5.pth'
    if os.path.exists(classifier_path):
        classifier = LeNet5()
        classifier.load_state_dict(torch.load(classifier_path))
        print(f"加载已保存的 LeNet-5 分类器从 {classifier_path}")
    else:
        classifier = train_lenet5_classifier(train_dataset_full, model_path=classifier_path)
    
    evaluate_classifier(classifier, test_dataset, "LeNet-5")
    classifier.eval()
    
    print("为 ScoreNets 按类别准备数据集...")
    indices_per_class = {i: [] for i in range(10)}
    for i, label in enumerate(train_dataset_full.targets):
        indices_per_class[label.item()].append(i)

    train_datasets_per_class = {
        label: Subset(train_dataset_full, indices)
        for label, indices in indices_per_class.items()
    }
    
    score_nets = train_score_networks(train_datasets_per_class)
    for net in score_nets.values(): net.eval()
    
    D_counts = {i: len(indices) for i, indices in indices_per_class.items()}

    success_count = 0
    fail_count = 0
    skipped_count = 0

    for instance_index in instance_indices:
        print("-" * 50)
        print(f"正在处理索引为 {instance_index} 的实例...")
        
        x0, true_label = test_dataset[instance_index]
        x0 = x0.unsqueeze(0)
        
        pred_label = classifier(x0).argmax().item()
        print(f"实例信息: 真实标签 = {true_label}, 分类器预测 = {pred_label}")

        if pred_label != true_label:
            print("警告：分类器对原始图像预测错误，跳过此实例。")
            skipped_count += 1
            continue
        
        target_label = algorithm1(x0, classifier, score_nets, D_counts, alpha=params['alpha'])

        if target_label is not None:
            print(f"Algorithm 1 找到目标: {target_label}. 开始 Algorithm 2...")
            counterfactual_image = algorithm2(
                x0, target_label, classifier, score_nets, D_counts,
                beta=params['beta'], eta=params['eta'], gamma=params['gamma'], delta=params['delta']
            )
            
            with torch.no_grad():
                output = classifier(counterfactual_image)
                final_pred = output.argmax().item()

                if final_pred == target_label:
                    print(f"*** 攻击成功! 原始: {pred_label} -> 最终: {final_pred} (目标: {target_label}) ***")
                    success_count += 1
                else:
                    print(f"--- 攻击失败。原始: {pred_label} -> 最终: {final_pred} (目标: {target_label}) ---")
                    fail_count += 1
            
            # <--- 修改5：在批量测试中，通常不为每个样本都保存图片，故注释掉
            # 如果需要保存某个成功的样本，可以手动取消注释
            # save_path = os.path.join(output_dir, f"result_index_{instance_index}.png")

        else:
            print("--- 攻击失败。Algorithm 1 未能找到目标标签。 ---")
            fail_count += 1
            
    print("=" * 60)
    print("批量攻击测试完成！")
    print("-" * 20 + " 结果报告 " + "-" * 20)
    total_processed = success_count + fail_count
    print(f"总共测试样本数: {num_instances_to_test}")
    print(f"分类器原始预测错误的样本数 (已跳过): {skipped_count}")
    print(f"总有效攻击尝试次数: {total_processed}")
    print(f"  - 成功攻击次数: {success_count}")
    print(f"  - 失败攻击次数: {fail_count}")

    if total_processed > 0:
        success_rate = (success_count / total_processed) * 100
        print(f"\n攻击成功率: {success_rate:.2f}%")
    else:
        print("\n没有有效的攻击尝试，无法计算成功率。")
    print("=" * 60)


if __name__ == "__main__":
    main()

