## DFO-S 算法复现 （下载数据集->加载并训练->确认类别->优化->引入贝叶斯优化）##

# 下载并加载MNIST数据集，并定义一个简单的图像分类器
# 定义得分网络s_θ(·)
# 训练一个分类器模型
# 为每个类别训练一个得分网络
# 计算得分合成场
## 在算法1和算法2的基础下尝试实现DFO
# 算法1：使用得分合成技术识别目标类别
# 定义DFO-S的目标函数φ(x)
# DFO梯度近似
# 算法2： DFO-S：在得分合成场约束下生成图像反事实

# 创建环境（Linux）
conda create -n dfos python=3.10 -y
source ~/.bashrc   
conda init bash
conda activate dfos
# 下载相关数据包
pip install -r requirements.txt
## 在MNIST数据集下实现DFO-S
# 加载MNIST数据集，可根据下述代码转换为其他数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # 这里不需要在终端运行
# 准备得分网络
# 运行算法1，寻找目标类别
# 运行算法2，生成反事实图像
python3 dfo_s_mnist_main.py # 最初代码版本，后续已修改，这俩个运行module部分不兼容，跳过
# 需注意的是，在Linux服务器上通常没有GUI（图形界面），因而可视化时调用plt.show()会有无图像生成的情形。
python3 dfo_s_mnist_picture.py # 这里重新整了个可在autodl上生成图像文件的适配版本，也和后边更新的行module部分不兼容，跳过

输出：
加载 MNIST 数据集...
开始训练分类器...
分类器训练完成。

准备得分网络...
训练类别 0 的得分网络...
训练类别 1 的得分网络...
训练类别 2 的得分网络...
训练类别 3 的得分网络...
训练类别 4 的得分网络...
训练类别 5 的得分网络...
训练类别 6 的得分网络...
训练类别 7 的得分网络...
训练类别 8 的得分网络...
训练类别 9 的得分网络...

运行算法1：寻找目标类别...
原始标签: 7
/root/autodl-tmp/DFO/dfo_s_modules.py:101: RuntimeWarning: overflow encountered in exp
  weights[label] = (np.exp(count) + 1) / (sum([np.exp(D_counts[l]) for l in other_labels]) + len(other_labels) - 1)
/root/autodl-tmp/DFO/dfo_s_modules.py:101: RuntimeWarning: invalid value encountered in scalar divide
  weights[label] = (np.exp(count) + 1) / (sum([np.exp(D_counts[l]) for l in other_labels]) + len(other_labels) - 1)
找到目标标签: 0

运行算法2：生成反事实图像...
反事实图像已生成。
结果已保存到 ./results/counterfactual_from_7_to_0.png

# 观察生成的png图像，我们所做的事情首先是为mnist数据集中的每个类别0-9分布训练一个ScoreNet，然后挑选原始标签7.并为其找到一个反事实目标。
# 输出结果显示为找到目标标签0（算法1确认类别），但在运行生成反事实图像时，发现输出的图像被未知错误覆盖，并未生成即像7又像数字0的图片
# 找到输出这一段：RuntimeWarning: overflow encountered in exp和invalid value encountered in scalar divide，查验后发现在计算 np.exp(count) 时，由于数据集中的样本数量 count 很大，导致 e^count 的值超出了浮点数的表示范围，产生了溢出。虽然这通常不会导致程序崩溃，但它会使 weights 变量变为 inf 或 nan（不是一个数），从而可能影响得分合成场的计算。
# 需对count做归一化处理或使用logsumexp等数值计算方法。

# 新增网络架构生成（和后续修改不兼容，跳过）
sudo apt-get install graphviz

python3 networks.py # SimpleCNN及ScoreNet网络架构

# 用git命令下载PlotNeuralNet项目（可选）
sudo apt-get update
sudo apt-get install git
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git

python3 generate_config.py
# 生成论文所需latex图片(已失效)
python3 PlotNeuralNet/pyexamples/NN_SVG.py .config/simple_cnn_config.json > simple_cnn_diagram.tex
python3 PlotNeuralNet/pyexamples/NN_SVG.py .config/scorenet_config.json > scorenet_diagram.tex

## 新途径(仍是SimpleCNN及ScoreNet的网络架构) 
## 这一部分需执行
sudo apt-get update
sudo apt-get install graphviz

pip install torchviz

python3 visualize_models.py # 可选，生成网络架构图

# 可视化实例
python run_modules.py -i 0 1 2 3 4 5 6 7 8 9 # 添加想要处理的样本
python run_modules.py --visualize_pca 0 # PCA可视化
python run_modules.py --visualize_quiver 0 # 流场箭头图


# 探讨
2025.9.12
模型权重+ScoreNet保存（检查是否已有预训练等结果）
梳理所有出现的超参数（用word文档统计，调试，明白作用及取值范围），这里是MNIST数据集
让代码生成效果是一个数字图（任务）
保存原始数据到另一个标签中的中间数据


2025.9.18
模型运行速度问题：dfo_gradient低效循环（for循环太多）
1000 (algorithm2 迭代) * 150 (dfo_gradient 迭代) = 150,000 次
优化方式：用Pytorch张量操作批处理

在找到目标标签时，得到这个目标标签的值并输出，再以它为优化起点进行优化，置信度变低

# 一百个实例调用运行
python run_modules.py -i 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99


2025.10.10
1. 超参数整理：梳理出现的所有超参数，形成word/md文档，总结每个超参数的作用，取值范围(for MINIST)。挑参数：速度加快，可视化效果更好。
2. 场的可视化，每个场一个图，合成的场一个图; 箭头颜色改成箭头长短
3. 扰动可视化中，红蓝换一下
4. 增加优化迭代次数，调整参数，可视化效果更加好，尝试 CiFar-10

## 调整（如果需要调整超参数重新评估，需要执行下述命令）
# 重新运行测试时，需要执行下述命令删除之前训练的score_nets
rm -rf ./score_nets  # 可观察num_epochs10-50和gamma0.01-0.5的调整成功率变化

模型训练超参数 (LeNet-5 & ScoreNet)
| 超参数 | 所在函数 | 作用 | 取值范围 (MNIST) | 备注 |
| :----- | :----- | :----- | :----- | :----- |
| **num_epochs** | train_lenet5_classifier | LeNet-5 分类器模型训练的总轮次。 | 10 ~ 20 | 影响模型的最终准确率。 |
| **lr (学习率)** | train_lenet5_classifier | Adam 优化器学习率。 | 1e-3 |用于更新分类器参数。 |
| **batch_size** | train_lenet5_classifier | LeNet-5 训练时每批次样本数。 | 64 ~ 128 | 影响训练速度和内存占用。 |
| **sigma** (σ) | train_score_networks | 训练 ScoreNet 时添加到样本的噪声标准差，是 Score Matching 损失的关键参数。 | 0.01 ~ 0.05 | 影响 ScoreNet 估计数据分布梯度的精度。 |
| **num_epochs** | train_score_networks | ScoreNet 模型训练的总轮次。 | 10 ~ 50 | ScoreNet 通常需要比分类器更多的训练轮次。 |
| **lr (学习率)** | train_score_networks | Adam 优化器学习率。 | 1e-3 |用于更新得分网络参数。 |
| **batch_size** | train_score_networks | train_score_networks 训练时每批次样本数。 | 32 ~ 64 | |

算法迭代与步长控制
| 超参数 | 算法/函数 | 作用 | 取值范围 (MNIST) | 备注 |
| :----- | :----- | :----- | :----- | :----- |
| **alpha** (α) | algorithm1 | **Score Synthesis Field 驱动步长。** 控制在 Score Field 驱动下，寻找目标标签时的迭代步长。 | 0.1 ~ 1.0 | Algorithm 1 的关键参数，用于快速定位目标区域。 |
| **eta** (η) | algorithm2 | **梯度下降学习率。** 控制每次迭代中，对抗样本向 C&W 损失梯度方向移动的步长。 | 0.1 ~ 2.0 | Algorithm 2 中主要的优化步长。 |
| **gamma** (γ) | algorithm2 | **Score Field 权重系数。** 控制 Score Synthesis Field 对梯度下降的修正和正则化作用强度。 | 0.01 ~ 0.1 | 平衡梯度驱动（攻击）和得分驱动（分布保持）。 |
| **迭代次数** | algorithm1 | 寻找目标标签的最大迭代次数。 | 500 | 一旦找到目标标签，立即停止。 |
| **迭代次数** | algorithm2 | DFO-S 算法的最大迭代次数。 | 1000 ~ 3000 | 影响找到最优对抗样本的时间和精度。 |

目标函数与梯度估计
| 超参数 | 算法/函数 | 作用 | 取值范围 (MNIST) | 备注 |
| :----- | :----- | :----- | :----- | :----- |
| **beta** (β) | algorithm2, phi_cw | **距离正则化项系数。** 用于平衡 C&W 损失和 L2 距离。 | 1e-4 ~ 1e-2 | 值越小，对扰动大小的惩罚越轻，L2 距离可能越小。 |
| **delta** (δ) | dfo_gradient | **DFO 扰动半径。** 控制用于估计梯度的随机扰动大小。 | 1e-3 ~ 1.0 | 影响梯度估计的精度和方差。 |
| **N** | dfo_gradient | **DFO 采样方向数。** DFO 梯度估计中采样的随机方向数。 | 100 ~ 200 | 值越大估计越精确，但计算开销越大。 |
| **kappa** (κ) | phi_cw | **C&W 置信度参数。** 用于控制目标类别 Logit 与最高非目标类别 Logit 之间的间隔。 | 0 (或 0 ~ 100) | 这里固定为 0，表示只要目标 Logit 大于其它 Logit 即可。 |

2025.10.24
# Cifar10数据集测试
改动：CIFAR10是32x32彩色图像(3通道)，MNIST的代码时28x28的灰度图像(1通道)
1.LeNet5 架构：修改输入通道、卷积层和全连接层尺寸
2.ScoreNet 架构：修改输入和输出尺寸
3.数据加载和归一化：切换数据集并调整归一化参数

运行测试：
python cifar10_run_modules.py -i 0 1 2 # 可添加更多案例

python cifar10_run_modules.py --visualize_pca 0

python cifar10_run_modules.py --visualize_quiver 0

0.airplane (飞机)
1.automobile (汽车)
2.bird (鸟)
3.cat (猫)
4.deer (鹿)
5.dog (狗)
6.frog (青蛙)
7.horse (马)
8.ship (船)
9.truck (卡车)

# 同MINIST调整，重新运行测试时，需要执行下述命令删除之前训练的lenet5和score_nets
# MINIST只需调整score_nets是因为lenet5训练的还可以
rm -f cifar10_lenet5.pth
rm -rf ./cifar10_score_nets


## 总结（Linux环境下运行简化版）
conda create -n dfos python=3.10 -y
source ~/.bashrc   
conda init bash    
conda activate dfos

pip install -r requirements.txt

# MINIST
python run_modules.py -i 0 1 2 
python run_modules.py --visualize_pca 0 
python run_modules.py --visualize_quiver 0 

rm -rf ./score_nets # 不重新预训练无需执行

# Cifar-10
python cifar10_run_modules.py -i 0 1 2 
python cifar10_run_modules.py --visualize_pca 0
python cifar10_run_modules.py --visualize_quiver 0

rm -f cifar10_lenet5.pth # 不重新预训练无需执行
rm -rf ./cifar10_score_nets # 不重新预训练无需执行