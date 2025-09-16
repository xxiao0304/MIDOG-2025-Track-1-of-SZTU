import os
import json
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import time

# 全局变量存储当前时间戳（确保同一轮训练中时间戳一致）
_TIMESTAMP = None

def create_output_dirs(base_dir="class"):
    """创建带时间戳的输出目录结构，仍返回基础目录字符串"""
    global _TIMESTAMP
    # 生成唯一时间戳（同一轮训练中只生成一次）
    if _TIMESTAMP is None:
        _TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    
    # 创建基础目录
    os.makedirs(base_dir, exist_ok=True)
    
    # 创建带时间戳的子目录
    os.makedirs(os.path.join(base_dir, f"models_{_TIMESTAMP}"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f"metrics_{_TIMESTAMP}"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "predictions"), exist_ok=True)
    
    # 仍返回基础目录（保持与主代码调用方式一致）
    return base_dir, _TIMESTAMP

def get_model_dir(base_dir):
    """获取带时间戳的模型目录路径"""
    global _TIMESTAMP
    if _TIMESTAMP is None:
        raise ValueError("请先调用 create_output_dirs 初始化目录")
    return os.path.join(base_dir, f"models_{_TIMESTAMP}")

def get_metric_dir(base_dir):
    """获取带时间戳的指标目录路径"""
    global _TIMESTAMP
    if _TIMESTAMP is None:
        raise ValueError("请先调用 create_output_dirs 初始化目录")
    return os.path.join(base_dir, f"metrics_{_TIMESTAMP}")

def save_metrics(metrics, output_dir="class/metrics", prefix=""):
    """保存评估指标到带时间戳的目录（兼容原调用方式）"""
    # 自动替换为带时间戳的指标目录
    global _TIMESTAMP
    if _TIMESTAMP is not None:
        # 从基础目录中提取前缀，拼接时间戳目录
        base_dir = os.path.dirname(output_dir) if "metrics_" not in output_dir else output_dir
        output_dir = get_metric_dir(base_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为JSON
    json_path = os.path.join(output_dir, f"{prefix}metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 保存为TXT
    txt_path = os.path.join(output_dir, f"{prefix}metrics.txt")
    with open(txt_path, 'w') as f:
        f.write("===== 评估指标汇总 =====\n\n")
        f.write(f"准确率: {metrics['accuracy']:.4f}\n")
        f.write(f"有效样本数: {metrics['total_valid_samples']}\n")
        f.write(f"无效样本数: {metrics['total_invalid_samples']}\n\n")
        
        f.write("类别指标:\n")
        f.write("{:<15} {:<10} {:<10} {:<10} {:<10}\n".format(
            "类别", "精确率", "召回率", "F1分数", "支持样本数"))
        for i, class_name in enumerate(metrics['class_names']):
            f.write("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}\n".format(
                class_name, 
                metrics['precision'][i], 
                metrics['recall'][i], 
                metrics['f1'][i], 
                metrics['support'][i]))
        
        f.write("\n混淆矩阵:\n")
        cm = np.array(metrics['confusion_matrix'])
        f.write("真实\\预测 " + " ".join([f"{c:<10}" for c in metrics['class_names']]) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{metrics['class_names'][i]:<10} " + " ".join([f"{x:<10}" for x in row]) + "\n")

def save_checkpoint(model, optimizer, epoch, val_loss, save_path="class/models/best_model.pth"):
    """保存模型检查点到带时间戳的目录（兼容原调用方式）"""
    # 自动替换为带时间戳的模型目录
    global _TIMESTAMP
    if _TIMESTAMP is not None:
        # 提取基础路径和文件名
        base_dir = os.path.dirname(save_path)
        filename = os.path.basename(save_path)
        # 拼接带时间戳的路径
        save_path = os.path.join(get_model_dir(base_dir), filename)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'augmentation_params': {
            'mixup_alpha': 0.2,
            'cutmix_alpha': 0.2,
            'mix_prob': 0.5
        }
    }, save_path)
    return save_path

def evaluate_model(model, dataloader, device, class_names, filter_invalid=True):
    """评估模型性能（保持不变）"""
    model.eval()
    all_preds = []
    all_labels = []
    total_invalid_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if filter_invalid:
                valid_mask = labels != -1
                total_invalid_samples += (valid_mask == False).sum().item()
                inputs = inputs[valid_mask]
                labels = labels[valid_mask]
                if len(labels) == 0:
                    continue
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                cls_output = outputs[0]
            else:
                cls_output = outputs
            
            _, preds = torch.max(cls_output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    total_valid_samples = len(all_labels)
    if total_valid_samples == 0:
        return {
            'accuracy': 0.0,
            'precision': [0.0]*len(class_names),
            'recall': [0.0]*len(class_names),
            'f1': [0.0]*len(class_names),
            'support': [0]*len(class_names),
            'confusion_matrix': [[0]*len(class_names) for _ in range(len(class_names))],
            'class_names': class_names,
            'total_valid_samples': 0,
            'total_invalid_samples': total_invalid_samples
        }
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names)))).tolist()
    
    return {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm,
        'class_names': class_names,
        'total_valid_samples': total_valid_samples,
        'total_invalid_samples': total_invalid_samples
    }
