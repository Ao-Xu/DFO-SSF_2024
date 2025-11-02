import json
import os

# 配置 SimpleCNN 模型的架构
cnn_config = {
    "layers": [
        {"title": "输入层", "id": "input", "type": "input", "label": "MNIST图像", "units": "28x28", "color": "darkblue"},
        {"title": "卷积层1", "id": "conv1", "type": "conv", "units": 10, "kernel": "5x5", "stride": 1, "padding": 0, "color": "orange"},
        {"title": "最大池化层", "id": "pool1", "type": "pool", "units": 10, "kernel": "2x2", "stride": 2, "color": "red"},
        {"title": "卷积层2", "id": "conv2", "type": "conv", "units": 20, "kernel": "5x5", "stride": 1, "padding": 0, "color": "orange"},
        {"title": "最大池化层", "id": "pool2", "type": "pool", "units": 20, "kernel": "2x2", "stride": 2, "color": "red"},
        {"title": "展平层", "id": "flatten", "type": "flatten", "units": 320, "color": "green"},
        {"title": "全连接层1", "id": "fc1", "type": "dense", "units": 50, "color": "green"},
        {"title": "全连接层2", "id": "fc2", "type": "dense", "units": 10, "color": "green"},
        {"title": "输出层", "id": "output", "type": "softmax", "units": 10, "color": "pink", "label": "输出概率"}
    ],
    "connections": [
        {"from": "input", "to": "conv1", "style": "solid", "label": ""},
        {"from": "conv1", "to": "pool1", "style": "solid", "label": ""},
        {"from": "pool1", "to": "conv2", "style": "solid", "label": ""},
        {"from": "conv2", "to": "pool2", "style": "solid", "label": ""},
        {"from": "pool2", "to": "flatten", "style": "solid", "label": ""},
        {"from": "flatten", "to": "fc1", "style": "solid", "label": ""},
        {"from": "fc1", "to": "fc2", "style": "solid", "label": ""},
        {"from": "fc2", "to": "output", "style": "solid", "label": ""}
    ]
}

# 配置 ScoreNet 模型的架构
scorenet_config = {
    "layers": [
        {"title": "输入层", "id": "input", "type": "input", "units": 784, "color": "darkblue", "label": "展平后的图像"},
        {"title": "全连接层1", "id": "fc1", "type": "dense", "units": 512, "color": "green"},
        {"title": "全连接层2", "id": "fc2", "type": "dense", "units": 512, "color": "green"},
        {"title": "输出层", "id": "output", "type": "dense", "units": 784, "color": "pink", "label": "得分估计"},
    ],
    "connections": [
        {"from": "input", "to": "fc1", "style": "solid", "label": ""},
        {"from": "fc1", "to": "fc2", "style": "solid", "label": ""},
        {"from": "fc2", "to": "output", "style": "solid", "label": ""}
    ]
}

def generate_json_file(config, filename, output_dir=".config"):
    """
    将Python字典保存为JSON文件，并将其放入指定的输出目录。
    如果目录不存在，则自动创建。
    """
    # 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 组合完整的文件路径
    file_path = os.path.join(output_dir, filename)

    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"配置文件 '{filename}' 已成功保存到 '{output_dir}' 目录。")

if __name__ == "__main__":
    generate_json_file(cnn_config, "simple_cnn_config.json")
    generate_json_file(scorenet_config, "scorenet_config.json")
