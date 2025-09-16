import os
import torch
import argparse
import time
import torch.nn.functional as F
from dataset import create_datasets  # 需确保 dataset.py 路径正确
from model import MitosisClassifier  # 需确保 model.py 路径正确
from utils import evaluate_model, save_metrics  # 复用 utils 函数
import logging
from collections import defaultdict

# -------------------------- 配置日志 --------------------------
def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"test_{time.strftime('%Y%m%d-%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# -------------------------- 核心测试逻辑 --------------------------
def test_model(model_path, device="cuda"):
    logger = setup_logging()
    logger.info(f"开始测试模型：{model_path}")
    
    # 1. 加载数据集（仅测试集）
    _, _, test_dataset = create_datasets()  # 需确保 create_datasets 返回 (train, val, test)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset, batch_size=960, shuffle=False,
        num_workers=12, pin_memory=True
    )
    
    # 2. 初始化模型并加载权重
    model = MitosisClassifier(model_size="tiny", num_classes=2)  # 二分类
    model = model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 切换到评估模式
    
    # 3. 模型推理并收集结果（按图像分组）
    class_names = ["non-mitosis", "mitosis"]
    image_results = defaultdict(list)  # 键：图像ID，值：该图像的所有预测结果
    total_invalid = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # 过滤无效样本（标签=-1）
            valid_mask = labels != -1
            total_invalid += (valid_mask == False).sum().item()
            
            if not valid_mask.any():
                continue  # 跳过全无效样本批次
            
            # 处理有效样本
            valid_inputs = inputs[valid_mask].to(device)
            valid_labels = labels[valid_mask]
            
            # 模型推理
            outputs = model(valid_inputs)
            cls_output = outputs[0] if isinstance(outputs, tuple) else outputs
            confs = F.softmax(cls_output, dim=1)
            preds = torch.argmax(confs, dim=1)
            max_confs = torch.max(confs, dim=1).values
            
            # 获取有效样本在原始数据集中的索引
            batch_start = batch_idx * test_loader.batch_size
            valid_indices = [batch_start + i for i, is_valid in enumerate(valid_mask) if is_valid]
            
            # 收集结果（按图像分组）
            for idx, pred, conf in zip(valid_indices, preds.cpu().numpy(), max_confs.cpu().numpy()):
                sample = test_dataset.samples[idx]
                img_path = sample['img_path']
                img_id = os.path.splitext(os.path.basename(img_path))[0]  # 提取图像文件名（不含扩展名）
                
                # 计算归一化边界框坐标（xyhw格式）
                x1, y1, x2, y2 = sample['x1'], sample['y1'], sample['x2'], sample['y2']
                img_size = test_dataset.img_size  # 原始图像尺寸（512）
                x_center = (x1 + x2) / 2 / img_size
                y_center = (y1 + y2) / 2 / img_size
                width = (x2 - x1) / img_size
                height = (y2 - y1) / img_size
                
                # 格式化结果行（类别 归一化x 归一化y 归一化w 归一化h 置信度）
                result_line = f"{pred} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}"
                image_results[img_id].append(result_line)
    
    logger.info(f"测试集推理完成，有效样本数: {sum(len(v) for v in image_results.values())}, 无效样本数: {total_invalid}")
    
    # 4. 保存分类结果（每个图像一个txt文件）
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_base = "test_results"
    metrics_dir = os.path.join(output_base, f"metrics_{timestamp}")
    preds_dir = os.path.join(metrics_dir, "predictions")  # 预测结果子目录
    os.makedirs(preds_dir, exist_ok=True)
    
    # 按图像保存结果
    for img_id, lines in image_results.items():
        txt_path = os.path.join(preds_dir, f"{img_id}.txt")
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
    logger.info(f"分类结果已保存至：{preds_dir}（共{len(image_results)}个图像文件）")
    
    # 5. 保存常规评估指标
    test_metrics = evaluate_model(
        model, test_loader, device, class_names, filter_invalid=True
    )
    save_metrics(
        test_metrics, 
        output_dir=metrics_dir, 
        prefix=f"test_{os.path.basename(model_path).split('.')[0]}_"
    )
    logger.info(f"测试完成！指标已保存至：{metrics_dir}")

# -------------------------- 命令行参数解析 --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立测试模型脚本")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/home/jupyter-x/ultralytics-main/mitosis_classification/models/models_20250829-130509/best_model.pth", 
        help="模型文件路径（如：class/models_20250827-001724/best_model.pth）"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="设备选择（cuda/cpu）"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"错误：模型文件不存在 → {args.model_path}")
        exit(1)
    
    test_model(args.model_path, args.device)