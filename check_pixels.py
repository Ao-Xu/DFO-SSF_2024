import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime

def load_and_preprocess_image_from_png(image_path, target_size=(28, 28)):
    """
    加载由 matplotlib 保存的 PNG 图像，并将其缩放至目标尺寸。
    
    参数:
    image_path (str): PNG 图像的路径。
    target_size (tuple): 目标尺寸，默认为 (28, 28)。
    
    返回:
    torch.Tensor: 标准化后的图像 Tensor。
    """
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_pil = Image.open(image_path).convert('L')
    img_pil = img_pil.resize(target_size, Image.LANCZOS)
    
    return transform(img_pil).unsqueeze(0)

def visualize_and_analyze_perturbation(original_image_path, counterfactual_image_path, target_region, analysis_name, output_dir):
    """
    加载图像，计算扰动，可视化并分析指定区域的像素扰动，并将结果保存为图片。
    
    参数:
    original_image_path (str): 原始图像的路径 (.png)。
    counterfactual_image_path (str): 对抗样本的路径 (.png)。
    target_region (tuple): 感兴趣区域的坐标，格式为 (y_start, y_end, x_start, x_end)。
    analysis_name (str): 本次分析的名称，用于保存文件。
    output_dir (str): 保存结果图片的目录。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("-" * 50)
    print(f"开始像素扰动分析: {analysis_name}")
    print("-" * 50)
    
    try:
        x0_tensor = load_and_preprocess_image_from_png(original_image_path)
        counterfactual_tensor = load_and_preprocess_image_from_png(counterfactual_image_path)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。请确保文件路径正确。")
        return

    if x0_tensor.shape != counterfactual_tensor.shape:
        print("错误：重新采样后图像尺寸仍然不匹配。")
        return

    # --- 1. 计算扰动矩阵 ---
    perturbation = counterfactual_tensor - x0_tensor
    perturbation_np = perturbation.squeeze().detach().numpy()

    # --- 2. 提取并分析指定区域的像素值 ---
    y_start, y_end, x_start, x_end = target_region
    region_of_interest_tensor = perturbation[:, :, y_start:y_end, x_start:x_end]
    region_np = region_of_interest_tensor.squeeze().numpy()
    
    positive_pixels = np.sum(region_np > 0)
    negative_pixels = np.sum(region_np < 0)
    
    if positive_pixels > negative_pixels:
        conclusion_en = "Positive Perturbation Dominant"
        conclusion_zh = "正向扰动为主"
        text_color = "red"
    elif negative_pixels > positive_pixels:
        conclusion_en = "Negative Perturbation Dominant"
        conclusion_zh = "负向扰动为主"
        text_color = "blue"
    else:
        conclusion_en = "Balanced Perturbation"
        conclusion_zh = "正负扰动平衡"
        text_color = "green"

    # --- 3. 可视化扰动和感兴趣区域 ---
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    v_min, v_max = -0.3, 0.3
    plt.imshow(perturbation_np, cmap='RdBu', vmin=v_min, vmax=v_max)
    
    rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                         linewidth=2, edgecolor='yellow', facecolor='none')
    plt.gca().add_patch(rect)
    
    plt.text(x_start, y_start - 2, conclusion_en, color=text_color, fontsize=12, 
             weight='bold', bbox=dict(facecolor='white', alpha=0.7))

    plt.title('Perturbation (Red=Positive, Blue=Negative)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    region_of_interest_img = perturbation_np[y_start:y_end, x_start:x_end]
    plt.imshow(region_of_interest_img, cmap='RdBu', vmin=v_min, vmax=v_max)
    plt.title(f'Zoomed Region ({analysis_name})')
    plt.axis('off')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"analysis_result_{analysis_name}_{timestamp}.png")
    plt.savefig(save_path)
    print(f"分析结果图片已保存到 {save_path}")
    plt.close()

    print("\n" + "=" * 50)
    print("像素值分析报告")
    print(f"分析区域坐标: y={y_start}-{y_end}, x={x_start}-{x_end}")
    print("=" * 50)
    print("区域内的像素扰动值:")
    print(np.round(region_np, 4))
    
    print("\n统计信息:")
    print(f"  平均扰动值: {np.mean(region_np):.4f}")
    print(f"  最大扰动值: {np.max(region_np):.4f}")
    print(f"  最小扰动值: {np.min(region_np):.4f}")
    
    print(f"  正向扰动像素数: {positive_pixels}")
    print(f"  负向扰动像素数: {negative_pixels}")
    print(f"  总像素数: {region_np.size}")
    
    print(f"\n结论: 该区域以**{conclusion_zh}**为主。")
    print("-" * 50)


# --- 主程序 ---
if __name__ == '__main__':
    # 定义输入和输出路径
    input_dir = './results'
    output_dir = './analysis_results'
    
    original_image_path = os.path.join(input_dir, 'original_index_0.png')
    counterfactual_image_path = os.path.join(input_dir, 'SUCCESS_REPLICATION_index_0.png')
    
    # 分析关键区域
    # 区域1:下半部分曲线
    analysis_1_name = 'Lower-Curve-of-3'
    target_region_A = (15, 25, 0, 15)
    visualize_and_analyze_perturbation(original_image_path, counterfactual_image_path, target_region_A, analysis_1_name, output_dir)
    
    # 区域2:右下角竖线 
    analysis_2_name = 'Vertical-Line-of-7'
    target_region_B = (15, 25, 12, 22)  
    visualize_and_analyze_perturbation(original_image_path, counterfactual_image_path, target_region_B, analysis_2_name, output_dir)

    # 区域3:顶部连接处 
    analysis_3_name = 'Top-Connection-of-7'
    target_region_C = (5, 12, 10, 22)
    visualize_and_analyze_perturbation(original_image_path, counterfactual_image_path, target_region_C, analysis_3_name, output_dir)

    print("\n所有分析完成。请检查 'analysis_results' 目录下的图片文件。")





