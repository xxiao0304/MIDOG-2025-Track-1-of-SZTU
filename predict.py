import os
import json
import math
from PIL import Image
from ultralytics import YOLO

# 配置参数 - 仅需修改这些参数
model_path = "runs/train_single/exp_lsema/weights/best.pt"  # 模型权重路径
image_path = "007.tiff"  # 待处理的单张图像路径
output_dir = "predict_single"  # 输出目录
conf_threshold = 0.2    # 预测置信度阈值（过滤低置信度预测）
iou_threshold = 0.3     # NMS的IoU阈值
patch_size = 512        # 分块大小
overlap_ratio = 0.3     # 重叠度
class_threshold = 0.45  # 类别判断阈值（>0.45为mitotic figure）

# 计算实际重叠像素数和步长
overlap = int(patch_size * overlap_ratio)
step = patch_size - overlap

def split_image(image, patch_size=512, step=358):
    """将大图分成多个重叠的小块"""
    width, height = image.size
    patches = []
    coordinates = []
    
    # 计算需要多少块才能覆盖整个图像
    num_cols = math.ceil((width - patch_size) / step) + 1 if width > patch_size else 1
    num_rows = math.ceil((height - patch_size) / step) + 1 if height > patch_size else 1
    
    for i in range(num_rows):
        for j in range(num_cols):
            # 计算当前块的左上角坐标
            x = j * step
            y = i * step
            
            # 确保最后一块不会超出图像边界
            if x + patch_size > width:
                x = width - patch_size
            if y + patch_size > height:
                y = height - patch_size
                
            # 提取图像块
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            coordinates.append((x, y))
    
    return patches, coordinates

def convert_to_original_coordinates(patch_results, x_offset, y_offset, patch_size, original_width, original_height):
    """将图像块中的相对坐标转换为原图中的绝对坐标"""
    converted_results = []
    
    for box in patch_results.boxes:
        # 获取YOLO格式的边界框 (x_center, y_center, width, height) - 相对图像块的比例
        x_center_rel, y_center_rel, _, _ = box.xywhn[0].tolist()
        conf = box.conf[0].item()
        cls = box.cls[0].item()
        
        # 转换为图像块中的像素坐标
        x_center = x_center_rel * patch_size
        y_center = y_center_rel * patch_size
        
        # 转换为原图中的像素坐标
        x_center_orig = x_center + x_offset
        y_center_orig = y_center + y_offset
        
        # 确保坐标在原图范围内
        x_center_orig = max(0, min(x_center_orig, original_width))
        y_center_orig = max(0, min(y_center_orig, original_height))
        
        # 计算用于NMS的坐标
        w = box.xywh[0][2].item()  # 宽度（像素）
        h = box.xywh[0][3].item()  # 高度（像素）
        
        converted_results.append({
            "class": int(cls),
            "confidence": conf,
            "x_center": x_center_orig,
            "y_center": y_center_orig,
            # 计算左上角和右下角坐标，用于NMS
            "x1": x_center_orig - w/2,
            "y1": y_center_orig - h/2,
            "x2": x_center_orig + w/2,
            "y2": y_center_orig + h/2
        })
    
    return converted_results

def non_max_suppression(results, iou_threshold):
    """对所有图像块的预测结果进行非极大值抑制，去除重叠区域的重复检测"""
    if not results:
        return []
    
    # 按置信度降序排序
    results.sort(key=lambda x: x["confidence"], reverse=True)
    
    nms_results = []
    while results:
        # 取置信度最高的框
        current = results.pop(0)
        nms_results.append(current)
        
        # 计算与当前框的IoU，过滤掉重叠度高的框
        to_remove = []
        for i, result in enumerate(results):
            # 计算IoU
            x1 = max(current["x1"], result["x1"])
            y1 = max(current["y1"], result["y1"])
            x2 = min(current["x2"], result["x2"])
            y2 = min(current["y2"], result["y2"])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_current = (current["x2"] - current["x1"]) * (current["y2"] - current["y1"])
            area_result = (result["x2"] - result["x1"]) * (result["y2"] - result["y1"])
            union = area_current + area_result - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                to_remove.append(i)
        
        # 从后往前删除，避免索引问题
        for i in reversed(to_remove):
            results.pop(i)
    
    return nms_results

def save_json_results(results, save_path, image_name):
    """将结果保存为指定格式的JSON文件"""
    json_results = {
        "name": "Points of interest",
        "type": "Multiple points",
        "points": [],
        "version": {"major": 1, "minor": 0}
    }
    
    for result in results:
        # 根据置信度确定类别名称
        if result["confidence"] > class_threshold:
            class_name = "mitotic figure"
        else:
            class_name = "non mitotic figure"
        
        # 中心点坐标，z轴固定为0.0
        point = {
            "name": class_name,
            "point": [
                round(result["x_center"], 6),  # x坐标，保留6位小数
                round(result["y_center"], 6),  # y坐标，保留6位小数
                0.0                           # z坐标固定为0.0
            ],
            "probability": round(result["confidence"], 6)  # 保留6位小数
        }
        
        json_results["points"].append(point)
    
    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=4)

def process_single_image():
    """处理单张图像：分块、预测、合并结果、保存为JSON"""
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在 - {image_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    try:
        model = YOLO(model_path)
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return
    
    # 读取图像
    try:
        image = Image.open(image_path)
        original_width, original_height = image.size
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"成功加载图像: {image_name}, 尺寸: {original_width}x{original_height}")
    except Exception as e:
        print(f"图像加载失败: {str(e)}")
        return
    
    # 分块
    print(f"正在将图像分块为 {patch_size}x{patch_size} 大小，重叠度 {overlap_ratio}")
    patches, coordinates = split_image(image, patch_size, step)
    print(f"分块完成，共生成 {len(patches)} 个图像块")
    
    # 处理所有图像块
    all_results = []
    for i, (patch, (x, y)) in enumerate(zip(patches, coordinates)):
        # 预测
        results = model(patch, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # 转换坐标到原图
        converted = convert_to_original_coordinates(results[0], x, y, patch_size, original_width, original_height)
        all_results.extend(converted)
        
        # 打印分块处理进度
        if (i + 1) % 5 == 0 or (i + 1) == len(patches):
            print(f"已处理 {i + 1}/{len(patches)} 个图像块")
    
    # 对所有结果进行NMS
    print("正在进行非极大值抑制去除重复检测...")
    nms_results = non_max_suppression(all_results, iou_threshold)
    print(f"检测完成，共发现 {len(nms_results)} 个目标")
    
    # 保存为json
    json_path = os.path.join(output_dir, f"{image_name}.json")
    save_json_results(nms_results, json_path, image_name)
    print(f"结果已保存至: {json_path}")

if __name__ == "__main__":
    process_single_image()
    print("所有任务已完成！")
