import torch
import torch.nn as nn
from torchviz import make_dot
import os

# 1. 定义 SimpleCNN 模型 (分类器 f(·))
# 该模型用于图像分类
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 定义 ScoreNet 模型 (得分网络 s_θ(·))
# 该模型用于估计数据分布的梯度
class ScoreNet(nn.Module):
    def __init__(self):
        super(ScoreNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784)
        )

    def forward(self, x):
        # 将图像展平为一维向量
        x = x.view(-1, 784)
        return self.fc(x).view(-1, 1, 28, 28)

def visualize_network_architecture(model, input_size, filename):
    """
    可视化神经网络架构，并保存为图片。
    
    Args:
        model (nn.Module): 要可视化的 PyTorch 模型。
        input_size (tuple): 模型的输入张量形状（例如，(1, 1, 28, 28)）。
        filename (str): 输出文件名。
    """
    # 创建一个虚拟输入张量，以便生成计算图
    dummy_input = torch.randn(input_size)
    
    # 获取模型的输出
    output = model(dummy_input)
    
    # 使用 make_dot 生成计算图，并设置保存格式和路径
    # 使用 os.path.join 确保跨操作系统的路径兼容性
    output_dir = "network_diagrams"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    diagram = make_dot(output, params=dict(model.named_parameters()))
    diagram.render(os.path.join(output_dir, filename), format="png")
    
    print(f"网络架构图 '{filename}.png' 已成功保存到 '{output_dir}' 文件夹中。")

if __name__ == "__main__":
    # 可视化 SimpleCNN
    cnn_model = SimpleCNN()
    # MNIST 图像输入大小：(batch_size, channels, height, width)
    input_shape_cnn = (1, 1, 28, 28)
    visualize_network_architecture(cnn_model, input_shape_cnn, "simple_cnn_architecture")
    
    print("-" * 50)
    
    # 可视化 ScoreNet
    score_model = ScoreNet()
    # ScoreNet 接受的输入与 CNN 相同
    input_shape_score = (1, 1, 28, 28)
    visualize_network_architecture(score_model, input_shape_score, "score_net_architecture")

