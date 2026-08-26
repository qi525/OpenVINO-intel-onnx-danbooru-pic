import cv2
import numpy as np
import pandas as pd
import logging
import datetime
import os
import sys
import operator
import time
import platform
import subprocess
import json
from PIL import Image
import io
import shlex
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any
import re
from tqdm import tqdm

# ==========================
# 1. CUDA / ONNX Runtime
# ==========================
try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    ort = None
    print(f"[WARN] 无法导入 onnxruntime: {e}")

try:
    import onnx
except Exception as e:  # pragma: no cover
    onnx = None
    print(f"[WARN] 无法导入 onnx: {e}")

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ========== SimpleTagClassifier 类定义 ==========
class SimpleTagClassifier:
    """
    简化的标签分类器 - 两层分类
    
    第一层分类规则：
      - 在标签列表中查找 FIRST_LEVEL_KEYWORDS
      - 按优先级顺序匹配，第一个匹配的作为第一层
      - 如果都不匹配，则为 'uncategorized'
    
    第二层分类规则：
      - 选择权重最高的标签
      - 排除黑名单中的标签
      - 排除第一层已分类的标签
      - 如果没有有效标签，则为 'uncategorized'
    """
    
    # ========== 配置参数 ==========
    # 第一层分类关键词及优先级顺序
    FIRST_LEVEL_KEYWORDS: List[str] = [
        'multiple_girls',  # 优先级 1 - 多个女性
        '2girls',          # 优先级 2 - 两个女性
        '1girl',           # 优先级 3 - 一个女性
        'no_humans',       # 优先级 4 - 无人类
    ]
    
    # 固定黑名单 - 第二层分类时排除的标签
    BLACKLIST_TAGS: Set[str] = {
        'general',         # 通用标签
        'generals',        # 通用标签（复数）
        'window',          # 特殊处理（可能包含禁用字符）
    }
    
    # Windows 文件系统禁用字符
    INVALID_CHARS_PATTERN = r'[<>:"/\\|?*]'
    
    def __init__(self, debug: bool = False, confidence_threshold: float = 0.5):
        """
        初始化分类器
        
        Args:
            debug: 是否启用调试输出
            confidence_threshold: 置信度阈值，用于第二层分类时过滤标签
        """
        self.debug = debug
        self.confidence_threshold = confidence_threshold
        logging.info(f"✓ 初始化标签分类器 v2")
        logging.info(f"  - 第一层关键词: {', '.join(self.FIRST_LEVEL_KEYWORDS)}")
        logging.info(f"  - 黑名单: {', '.join(sorted(self.BLACKLIST_TAGS))}")
        logging.info(f"  - 置信度阈值: {confidence_threshold}")
    
    @staticmethod
    def _is_valid_folder_name(name: str) -> bool:
        """
        检查文件夹名称是否有效（不包含 Windows 禁用字符）
        
        Args:
            name: 文件夹名称
        
        Returns:
            True 如果有效，False 如果包含禁用字符
        """
        if not name or not isinstance(name, str):
            return False
        return not bool(re.search(SimpleTagClassifier.INVALID_CHARS_PATTERN, name))
    
    def _classify_first_level(self, tags: List[str]) -> Tuple[str, str]:
        """
        第一层分类：核心主体识别（硬匹配）
        
        Args:
            tags: 小写标签列表
        
        Returns:
            (第一层分类结果, 分类说明)
        """
        for keyword in self.FIRST_LEVEL_KEYWORDS:
            if keyword in tags:
                return keyword, f"Contains '{keyword}'"
        return 'uncategorized', 'No first-level keyword matched'
    
    def _classify_second_level(self, tags: List[Tuple[str, float]], first_level: str) -> Tuple[str, str]:
        """
        第二层分类：特征语义提取（权重过滤）
        
        Args:
            tags: 从高到低排序的标签列表（包含权重信息）
                  格式：[(tag_name, confidence), ...]
            first_level: 第一层分类结果（用于排除）
        
        Returns:
            (第二层分类结果, 分类说明)
        """
        exclude_tags = self.BLACKLIST_TAGS | {first_level}
        
        for tag_name, confidence in tags:
            # ========== 置信度过滤 - 只考虑高于阈值的标签 ==========
            if confidence < self.confidence_threshold:
                if self.debug:
                    logging.debug(f"    [debug] 跳过置信度过低的标签: {tag_name} (confidence: {confidence:.4f} < {self.confidence_threshold})")
                continue
            
            tag_lower = tag_name.strip().lower()
            
            if tag_lower in exclude_tags:
                if self.debug:
                    logging.debug(f"    [debug] 跳过黑名单标签: {tag_lower}")
                continue
            
            if not self._is_valid_folder_name(tag_lower):
                if self.debug:
                    logging.debug(f"    [debug] 跳过包含禁用字符的标签: {tag_lower}")
                continue
            
            if self.debug:
                logging.debug(f"    [debug] 选择第二层标签: {tag_lower} (confidence: {confidence:.4f})")
            return tag_lower, f"Highest valid weight: {confidence:.4f}"
        
        return 'uncategorized', 'No valid second-level tag found'
    
    def classify(self, tags_list: List[Tuple[str, float]]) -> Tuple[str, str, dict]:
        """
        执行完整的两层分类
        
        Args:
            tags_list: 从高到低排序的标签列表
                      格式：[(tag_name, confidence), ...]
        
        Returns:
            (first_level, second_level, info_dict)
        """
        if not tags_list:
            return 'uncategorized', 'uncategorized', {
                'first_level': 'uncategorized',
                'second_level': 'uncategorized',
                'folder_path': 'uncategorized/uncategorized',
                'reason': 'Empty tag list',
                'raw_tags': [],
            }
        
        tag_names = [tag for tag, _ in tags_list]
        tags_lower = [tag.lower().strip() for tag in tag_names]
        
        first_level, first_reason = self._classify_first_level(tags_lower)
        second_level, second_reason = self._classify_second_level(tags_list, first_level)
        
        folder_path = f"{first_level}/{second_level}"
        full_reason = f"{first_reason} > {second_reason}"
        
        info = {
            'first_level': first_level,
            'second_level': second_level,
            'folder_path': folder_path,
            'reason': full_reason,
            'first_level_reason': first_reason,
            'second_level_reason': second_reason,
            'raw_tags': tag_names,
        }
        
        return first_level, second_level, info
    
    def get_folder_path(self, tags_list: List[Tuple[str, float]]) -> str:
        """获取完整的文件夹路径"""
        first_level, second_level, _ = self.classify(tags_list)
        return f"{first_level}/{second_level}"


def generate_txt_for_tags(image_path, tags_list, classifier):
    """
    为图片生成对应的 TXT 文件，用于训练数据
    TXT 文件名与图片文件名相同，只改后缀名
    TXT 内容为逗号分隔的标签列表
    
    Args:
        image_path: 图片文件路径
        tags_list: 标签列表，格式：[(tag_name, confidence), ...]
        classifier: SimpleTagClassifier 实例
    
    返回: (成功标志, TXT文件路径或错误信息, 分类信息)
    """
    try:
        # 检查源文件是否存在
        if not os.path.exists(image_path):
            logging.warning(f"生成TXT失败：文件不存在 - {image_path}")
            return False, "文件不存在", {}
        
        # 检查是否有有效标签
        if not tags_list:
            logging.debug(f"跳过生成TXT：{os.path.basename(image_path)} - 无有效标签")
            return False, "无有效标签", {}
        
        # 使用分类器进行两层分类
        first_level, second_level, class_info = classifier.classify(tags_list)
        
        # 提取所有tag名称并标准化
        all_tag_names = [tag[0].lower().strip() for tag in tags_list]
        
        # 第一步：过滤有效的tag（排除黑名单、禁用字符）
        valid_tags = []
        seen = set()
        for tag in all_tag_names:
            if tag not in seen:
                # 检查是否在黑名单中（要排除）
                if tag in classifier.BLACKLIST_TAGS:
                    logging.debug(f"    [filter] 跳过黑名单标签: {tag}")
                    continue
                # 检查是否包含禁用字符（要排除）
                if tag != 'uncategorized' and re.search(classifier.INVALID_CHARS_PATTERN, tag):
                    logging.debug(f"    [filter] 跳过包含禁用字符的标签: {tag}")
                    continue
                valid_tags.append(tag)
                seen.add(tag)
        
        # 第二步：确保first_level在最前面
        if first_level not in valid_tags:
            valid_tags.insert(0, first_level)
            seen.add(first_level)
        else:
            valid_tags.remove(first_level)
            valid_tags.insert(0, first_level)
        
        # 第三步：如果不足11个标签，从原始列表补充
        if len(valid_tags) < 11:
            logging.debug(f"    [补充] 有效tag仅{len(valid_tags)}个，需要补充{11 - len(valid_tags)}个")
            for tag in all_tag_names:
                if tag not in seen and len(valid_tags) < 11:
                    valid_tags.append(tag)
                    seen.add(tag)
            logging.debug(f"    [补充完成] 现有{len(valid_tags)}个tag")
        
        # 构造TXT文件路径（与图片同目录，只改后缀）
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        file_dir = os.path.dirname(image_path)
        txt_filename = f"{name_without_ext}.txt"
        txt_file_path = os.path.join(file_dir, txt_filename)
        
        # 如果TXT文件已存在，跳过（避免覆盖）
        if os.path.exists(txt_file_path):
            logging.debug(f"跳过生成TXT：文件已存在 - {txt_filename}")
            return True, txt_file_path, class_info  # 返回 True 表示此操作成功（已有TXT文件）
        
        # 将标签写入TXT文件（逗号分隔）
        tags_content = ', '.join(valid_tags)
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(tags_content)
        
        logging.info(f"成功生成TXT：{txt_filename} (分类: {class_info['reason']}, 标签数: {len(valid_tags)})")
        return True, txt_file_path, class_info
        
    except Exception as e:
        error_msg = f"生成TXT失败: {str(e)}"
        logging.error(f"生成TXT {image_path} 时发生错误: {error_msg}")
        return False, error_msg, {}


# --- 配置日志 ---
log_filename = f"onnx_cuda_inference_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

def setup_logging():
    """
    配置日志，确保在程序启动时生成新的日志文件。
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
    logging.info("日志系统初始化完成。")

setup_logging()


def print_step(step_no: int, title: str):
    """打印统一的流程步骤标题，确保每一步都在控制台输出。"""
    print(f"\n{'=' * 90}")
    print(f"[STEP {step_no}] {title}")
    print(f"{'=' * 90}")


def detect_cuda_environment() -> Dict[str, Any]:
    """
    检查当前机器是否具备 CUDA 运行环境，并把关键信息打印到控制台。
    主要检测内容：
      1) Python / OS / CPU 信息
      2) NVIDIA 驱动和 GPU 列表
      3) ONNX Runtime 可用 provider
      4) PyTorch CUDA 可用性（可选）
    """
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or "Unknown CPU",
        "nvidia_smi_ok": False,
        "gpu_names": [],
        "cuda_available": False,
        "cuda_provider": False,
        "torch_cuda_available": False,
        "providers": [],
        "gpu_count": 0,
    }

    print_step(1, "检查运行环境")
    print(f"Python 版本: {sys.version}")
    print(f"操作系统: {platform.platform()}")
    print(f"CPU: {info['cpu']}")
    print(f"架构: {platform.machine()}")

    # 检查 NVIDIA 驱动和显卡
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["nvidia_smi_ok"] = True
            gpu_lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
            info["gpu_names"] = gpu_lines
            info["gpu_count"] = len(gpu_lines)
            print(f"NVIDIA GPU 检测: 成功发现 {len(gpu_lines)} 张显卡")
            for i, line in enumerate(gpu_lines, 1):
                print(f"  GPU {i}: {line}")
        else:
            print("NVIDIA GPU 检测: 未发现可用的 nvidia-smi 输出，当前环境可能没有 NVIDIA 驱动或 GPU。")
            print(f"  nvidia-smi stderr: {result.stderr.strip()}")
    except Exception as e:
        print(f"NVIDIA GPU 检测异常: {e}")

    # 检查 ONNX Runtime provider
    if ort is not None:
        try:
            providers = ort.get_available_providers()
            info["providers"] = providers
            print(f"ONNX Runtime 可用 provider: {providers}")
            info["cuda_provider"] = "CUDAExecutionProvider" in providers
            info["cuda_available"] = info["cuda_provider"]
            print(f"CUDAExecutionProvider 可用: {'是' if info['cuda_provider'] else '否'}")
        except Exception as e:
            print(f"ONNX Runtime provider 检查失败: {e}")
    else:
        print("onnxruntime 未安装或导入失败，无法检测 provider。")

    # 检查 PyTorch CUDA
    if torch is not None:
        try:
            info["torch_cuda_available"] = torch.cuda.is_available()
            print(f"PyTorch CUDA 可用: {'是' if info['torch_cuda_available'] else '否'}")
            if info["torch_cuda_available"]:
                print(f"PyTorch 当前 CUDA 设备数量: {torch.cuda.device_count()}")
                print(f"PyTorch 当前 CUDA 设备: {torch.cuda.get_device_name(0)}")
                info["cuda_available"] = True
        except Exception as e:
            print(f"PyTorch CUDA 检测失败: {e}")
    else:
        print("PyTorch 未安装，跳过 CUDA 设备检测。")

    # 额外确认：如果系统没有 NVIDIA GPU，但是 ONNX Runtime 有 CUDA provider，也可能是环境变量或在容器中的特殊配置
    if info["cuda_available"]:
        print("CUDA 环境检测结论: 该环境具备 CUDA 运行能力，后续将优先使用 CUDAExecutionProvider。")
    else:
        print("CUDA 环境检测结论: 当前环境未检测到可用 CUDA，后续将强制回退到 CPUExecutionProvider。")

    return info


def validate_model_file(model_path: str) -> bool:
    """检查模型文件是否存在、大小是否合理，并输出详细日志。"""
    print_step(2, "检查 ONNX 模型文件")
    print(f"模型路径: {model_path}")
    if not model_path or not os.path.exists(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        return False

    file_size = os.path.getsize(model_path)
    print(f"模型文件大小: {file_size} bytes")

    if file_size <= 0:
        print("[ERROR] 模型文件大小为 0，无法执行推理。")
        return False

    if onnx is not None:
        try:
            model = onnx.load(model_path)
            print(f"ONNX 模型合法性: 通过检查，graph nodes={len(model.graph.node)}")
        except Exception as e:
            print(f"[WARN] ONNX 文件格式检查失败，但不一定意味着模型无法运行: {e}")
    else:
        print("onnx 包未安装，跳过模型结构校验。")

    return True


def build_onnx_session(model_path: str, prefer_cuda: bool = True):
    """
    构建 ONNX Runtime 推理 session。对 CUDA 环境做优先选择：
      - 若检测到 CUDAExecutionProvider，则使用 CUDA + CPU fallback
      - 否则使用 CPUExecutionProvider
    """
    if ort is None:
        raise RuntimeError("onnxruntime 没有安装，无法初始化 ONNX Runtime session。")

    print_step(3, "构建 ONNX Runtime Session")
    print(f"尝试加载模型: {model_path}")

    providers = ["CPUExecutionProvider"]
    if prefer_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("检测到 CUDAExecutionProvider，已选择 CUDA 优先方案。")
    else:
        print("未检测到 CUDAExecutionProvider，将使用 CPU 模式。")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_names = [o.name for o in session.get_outputs()]

    print(f"Session 构建成功。providers={session.get_providers()}")
    print(f"模型输入名: {input_name}")
    print(f"模型输入 shape: {input_shape}")
    print(f"模型输出名: {output_names}")
    return session


def print_cuda_setup_guidance():
    """输出明确的 CUDA 环境安装建议，便于用户快速修复。"""
    print("\n=== CUDA 安装建议 ===")
    print("1) 确认系统已安装 NVIDIA 驱动并且 GPU 可见。")
    print("2) 确认 CUDA Toolkit 与显卡驱动兼容。")
    print("3) 重新安装 ONNX Runtime CUDA 版本：")
    print("   pip uninstall onnxruntime -y")
    print("   pip install onnxruntime-gpu")
    print("4) 如果仍然失败，检查 Python 环境是不是和系统 NVIDIA 驱动/库不匹配。")
    print("5) 可用 provider 列表必须包含 CUDAExecutionProvider。")
    print("======================\n")


# --- 全局配置变量 ---
BATCH_SIZE = 4
MODEL_EXPECTED_CHANNELS = 3
MAX_RETRY_ATTEMPTS = 3

def preprocess_image_single(image_path, target_height, target_width):
    """
    预处理单张图片以符合模型输入要求。
    """
    image = None
    pil_read_success = False
    error_message = ""

    try:
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            if not image_bytes:
                error_message = f"图片文件 {image_path} 读取为空。"
                logging.error(error_message)
                return None, error_message
            logging.debug(f"DEBUG: 成功读取 {image_path} 为字节流，长度: {len(image_bytes)}")
        except FileNotFoundError:
            error_message = f"图片文件 {image_path} 未找到。"
            logging.error(error_message)
            return None, error_message
        except Exception as e:
            error_message = f"读取图片文件 {image_path} 失败：{e}"
            logging.error(error_message)
            return None, error_message

        raw_image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(raw_image_array, cv2.IMREAD_COLOR)

        if image is None:
            logging.warning(f"图片 {image_path} OpenCV解码失败。尝试使用PIL。")
            try:
                pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                image = np.array(pil_img)
                pil_read_success = True
                logging.debug(f"DEBUG: {image_path} 通过PIL成功读取。形状: {image.shape}, 类型: {image.dtype}")
            except Exception as pil_e:
                error_message = f"图片 {image_path} PIL读取失败：{pil_e}"
                logging.error(error_message)
                image = None

        if image is None:
            if not error_message:
                error_message = f"无法读取图片：{image_path} (PIL和OpenCV均无法读取)"
            logging.error(error_message)
            return None, error_message

        if not pil_read_success:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                logging.debug(f"DEBUG: {image_path} (OpenCV读取) 转换为RGB。")
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                logging.debug(f"DEBUG: {image_path} (OpenCV读取) 灰度图转换为RGB。")
            else:
                error_message = f"图片 {image_path} 具有不支持的通道数或形状 ({image.shape})，无法转换为RGB。"
                logging.error(error_message)
                return None, error_message

        original_height, original_width = image.shape[0], image.shape[1]
        logging.debug(f"DEBUG: {image_path} 原始尺寸: H={original_height}, W={original_width}. 目标尺寸: H={target_height}, W={target_width}.")

        image = cv2.resize(image, (target_width, target_height))
        image = image.astype(np.float32) / 255.0
        image = image.transpose((2, 0, 1))
        logging.debug(f"DEBUG: {image_path} 预处理完成。最终形状: {image.shape}, 数据范围: [{np.min(image):.4f}, {np.max(image):.4f}]")

        return image, None
    except Exception as e:
        error_message = f"图片预处理失败：{image_path}，错误：{e}"
        logging.error(error_message)
        return None, error_message

def preprocess_batch_images(image_paths, target_height, target_width):
    """
    预处理一个批次的图片。
    """
    preprocessed_images_successful = []
    successful_flags_for_all_inputs = []
    preprocess_errors_for_all_inputs = []

    for image_path in image_paths:
        single_image, error_msg = preprocess_image_single(image_path, target_height, target_width)
        if single_image is not None:
            preprocessed_images_successful.append(single_image)
            successful_flags_for_all_inputs.append(True)
            preprocess_errors_for_all_inputs.append(None)
        else:
            successful_flags_for_all_inputs.append(False)
            preprocess_errors_for_all_inputs.append(error_msg if error_msg else "未知预处理错误")

    if not preprocessed_images_successful:
        logging.warning("DEBUG: 批次中没有成功预处理的图片，无法创建批次张量。")
        return None, successful_flags_for_all_inputs, preprocess_errors_for_all_inputs

    batch_tensor = np.stack(preprocessed_images_successful, axis=0)

    logging.info(f"批次预处理完成。原始批次大小: {len(image_paths)}, 实际成功预处理图片: {len(preprocessed_images_successful)}。张量形状: {batch_tensor.shape}")
    return batch_tensor, successful_flags_for_all_inputs, preprocess_errors_for_all_inputs


def postprocess_output(output_data_for_single_image, labels, threshold=0.2):
    """
    后处理模型输出，获取预测标签和置信度。
    返回: (预测标签, 置信度, 带相似度的标签字符串, 纯标签字符串, 排序后的标签列表)
    """
    try:
        scores = 1 / (1 + np.exp(-output_data_for_single_image))

        candidate_tags_with_scores = []
        for idx, score in enumerate(scores):
            if score >= threshold:
                if idx < len(labels):
                    candidate_tags_with_scores.append((labels[idx], score))
                else:
                    logging.warning(f"标签索引 {idx} 超出 labels 列表范围 ({len(labels)})。")

        candidate_tags_with_scores.sort(key=operator.itemgetter(1), reverse=True)

        filtered_labels = [item[0] for item in candidate_tags_with_scores]
        filtered_scores = [item[1] for item in candidate_tags_with_scores]

        all_predicted_tags_with_confidence_str = ", ".join([f"{label} ({score:.4f})" for label, score in zip(filtered_labels, filtered_scores)])
        all_predicted_tags_pure_str = ", ".join(filtered_labels)

        predicted_label = filtered_labels[0] if filtered_labels else "无有效标签"
        confidence = filtered_scores[0] if filtered_scores else 0.0

        return predicted_label, confidence, all_predicted_tags_with_confidence_str, all_predicted_tags_pure_str, candidate_tags_with_scores

    except Exception as e:
        logging.error(f"后处理失败，错误：{e}")
        return "未知标签", 0.0, "后处理失败", "后处理失败", []

def load_labels_from_json(json_file_path):
    """
    从 JSON 文件中加载标签列表（用于新模型的 tag_mapping.json）。
    """
    try:
        import json
        with open(json_file_path, 'r', encoding='utf-8') as f:
            tag_mapping = json.load(f)
        
        # tag_mapping 的格式: {"0": {"tag": "general", "category": "Rating"}, ...}
        # 需要按索引顺序提取标签
        labels = []
        for idx in sorted([int(k) for k in tag_mapping.keys()]):
            tag_name = tag_mapping[str(idx)]["tag"]
            labels.append(tag_name)
        
        logging.info(f"成功从 JSON 文件加载 {len(labels)} 个标签。")
        return labels
    except FileNotFoundError:
        logging.error(f"标签文件未找到: {json_file_path}。")
        return []
    except Exception as e:
        logging.error(f"加载标签文件时发生错误：{e}")
        return []

def load_labels(labels_file_path, column_name=None):
    """
    从 CSV 文件中加载标签列表（保留以向后兼容）。
    """
    try:
        df = pd.read_csv(labels_file_path)

        if column_name:
            if column_name not in df.columns:
                raise ValueError(f"指定的列 '{column_name}' 不存在于 CSV 文件中。可用列: {df.columns.tolist()}")
            labels = df[column_name].tolist()
        else:
            labels = df.iloc[:, 0].tolist()
            logging.info(f"未指定标签列名，默认读取 CSV 文件的第一列作为标签。列名: '{df.columns[0]}'")

        labels = [str(label).strip() for label in labels if str(label).strip()]
        logging.info(f"成功从 CSV 文件加载 {len(labels)} 个标签。")
        return labels
    except FileNotFoundError:
        logging.error(f"标签文件未找到: {labels_file_path}。")
        return []
    except Exception as e:
        logging.error(f"加载标签文件时发生错误：{e}")
        return []

def load_paths_from_file(config_file: str) -> list:
    """从配置文件加载图片目录路径"""
    config_path = os.path.abspath(config_file)
    
    if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
        example_paths = [
            "# 在下方添加要处理的图片目录路径，每行一个",
            "# 例如:",
            r"C:\stable-diffusion-webui-reForge\outputs\txt2img-images\2026-03-28",
            r"D:\images\batch2",
            r"E:\Danbooru\sorted",
        ]
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(example_paths))
        logging.info(f"已创建配置文件: {config_path}")
        print(f"✓ 已创建配置文件: {config_path}")
    
    # 打开配置文件供用户编辑
    if sys.platform == "win32":
        try:
            os.startfile(config_path)
            print(f"\n📝 配置文件已打开: {config_path}")
            input("⏳ 编辑完成后，按 Enter 键继续...")
        except Exception as e:
            print(f"⚠ 无法打开配置文件编辑器: {e}")
            print(f"请手动编辑: {config_path}")
            input("编辑完成后按 Enter 键继续...")
    else:
        print(f"请编辑配置文件: {config_path}")
        input("编辑完成后按 Enter 键继续...")
    
    paths = []
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    paths.append(line)
    except Exception as e:
        logging.error(f"读取配置文件失败: {str(e)}")
        print(f"❌ 读取配置文件失败: {str(e)}")
    
    return paths


def get_user_confirmation(image_dirs: list) -> bool:
    """展示要处理的目录列表并获取用户确认"""
    print(f"\n{'='*60}")
    print(f"📂 将处理 {len(image_dirs)} 个目录:")
    print(f"{'='*60}")
    for idx, dir_path in enumerate(image_dirs, 1):
        print(f"  {idx}. {dir_path}")
    
    print(f"{'='*60}")
    confirm = input("\n✓ 确认处理？(y/n): ").strip().lower()
    return confirm == 'y'


def ask_yes_no(prompt: str, default: str = "no") -> bool:
    """询问用户是否选择 yes / no，支持默认值。"""
    default_value = default.strip().lower()
    if default_value not in {"yes", "no"}:
        default_value = "no"

    while True:
        answer = input(f"{prompt} ").strip().lower()
        if not answer:
            answer = default_value

        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print(f"[WARN] 请输入 yes 或 no。默认值为: {default_value}")


def export_csv_results(results_data: List[Dict[str, Any]], csv_path: str):
    """默认导出 CSV，统一使用两列：路径, tag。"""
    try:
        csv_rows = []
        for item in results_data:
            path_value = item.get("完整路径") or item.get("路径") or ""
            tag_value = item.get("所有预测标签") or item.get("预测标签") or ""
            csv_rows.append({"路径": path_value, "tag": tag_value})

        df = pd.DataFrame(csv_rows, columns=["路径", "tag"])
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[CSV] 默认已生成 CSV: {csv_path}")
        logging.info(f"默认 CSV 已生成: {csv_path}, 共 {len(df)} 行")
    except Exception as e:
        error_msg = f"导出 CSV 失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        logging.error(error_msg)


def rename_image_with_tags(image_path, tags_list, classifier):
    """
    重命名图片文件，添加标签信息到文件名
    格式: 原文件名@@@[first_level]_tag1_tag2_..._tag10.jpg
    
    Args:
        image_path: 图片文件路径
        tags_list: 标签列表，格式：[(tag_name, confidence), ...]
        classifier: SimpleTagClassifier 实例
    
    返回: (成功标志, 新文件名或错误信息, 新文件完整路径)
    """
    try:
        # 检查源文件是否存在
        if not os.path.exists(image_path):
            logging.warning(f"重命名失败：文件不存在 - {image_path}")
            return False, "文件不存在", None
        
        # 检查是否有有效标签
        if not tags_list:
            logging.debug(f"跳过重命名：{os.path.basename(image_path)} - 无有效标签")
            return False, "无有效标签", None
        
        # 使用分类器进行两层分类
        first_level, second_level, class_info = classifier.classify(tags_list)
        
        # 提取所有tag名称并标准化
        all_tag_names = [tag[0].lower().strip() for tag in tags_list]
        
        # 第一步：过滤有效的tag（排除黑名单、禁用字符，但保留uncategorized）
        valid_tags = []
        seen = set()
        for tag in all_tag_names:
            if tag not in seen:
                # 检查是否在黑名单中（要排除）
                if tag in classifier.BLACKLIST_TAGS:
                    logging.debug(f"    [filter] 跳过黑名单标签: {tag}")
                    continue
                # 检查是否包含禁用字符（要排除）
                if tag != 'uncategorized' and re.search(classifier.INVALID_CHARS_PATTERN, tag):
                    logging.debug(f"    [filter] 跳过包含禁用字符的标签: {tag}")
                    continue
                valid_tags.append(tag)
                seen.add(tag)
        
        # 第二步：确保first_level在最前面（必须包含，即使是uncategorized）
        if first_level not in valid_tags:
            # first_level不在有效tag中，强制插入到最前面
            valid_tags.insert(0, first_level)
            seen.add(first_level)
        else:
            # first_level已经在有效tag中，移到最前面
            valid_tags.remove(first_level)
            valid_tags.insert(0, first_level)
        
        # 第三步：凑够11个tag（如果不足则补充）
        unique_tags = valid_tags[:11]  # 先取前11个
        
        # 如果不足11个，从原始tag列表中补充（不再过滤黑名单/禁用字符，但要避免重复）
        if len(unique_tags) < 11:
            logging.debug(f"    [补充] 有效tag仅{len(unique_tags)}个，需要补充{11 - len(unique_tags)}个")
            for tag in all_tag_names:
                if tag not in seen and len(unique_tags) < 11:
                    unique_tags.append(tag)
                    seen.add(tag)
            logging.debug(f"    [补充完成] 现有{len(unique_tags)}个tag")
        
        # 构造新文件名
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        file_extension = os.path.splitext(filename)[1]
        
        # 如果已经包含@@@，说明是重复处理，需要提取原始名称
        if '@@@' in name_without_ext:
            name_without_ext = name_without_ext.split('@@@')[0]
        
        tags_str = '_'.join(unique_tags)
        new_filename = f"{name_without_ext}@@@{tags_str}{file_extension}"
        
        # 获取文件所在目录
        file_dir = os.path.dirname(image_path)
        new_file_path = os.path.join(file_dir, new_filename)
        
        # 如果新文件名与原文件名相同，说明不需要重命名
        if new_filename == filename:
            logging.debug(f"文件名无需更改：{filename}")
            return False, "文件名无需更改", new_file_path
        
        # 如果目标文件已存在，返回错误
        if os.path.exists(new_file_path):
            logging.warning(f"重命名失败：目标文件已存在 - {new_file_path}")
            return False, "目标文件已存在", new_file_path
        
        # 执行文件重命名
        os.rename(image_path, new_file_path)
        
        logging.info(f"成功重命名：{filename} → {new_filename} (分类: {class_info['reason']})")
        return True, new_filename, new_file_path
        
    except Exception as e:
        error_msg = f"重命名失败: {str(e)}"
        logging.error(f"重命名图片 {image_path} 时发生错误: {error_msg}")
        return False, error_msg, None

def organize_images_immediately(image_path, tags_list, classifier, base_dir=None):
    """
    推理完成后立即整理图片到对应的标签文件夹
    使用两层分类逻辑生成目标文件夹路径
    
    Args:
        image_path: 图片文件路径
        tags_list: 标签列表，格式：[(tag_name, confidence), ...]
        classifier: SimpleTagClassifier 实例
        base_dir: 基准目录，从image_paths.txt中指定。如果指定，目标文件夹将在base_dir下创建
    
    返回: (成功标志, 目标路径或None, 错误信息或None, 分类信息)
    """
    try:
        # 检查源文件是否存在
        if not os.path.exists(image_path):
            logging.warning(f"组织失败：文件不存在 - {image_path}")
            return False, None, "文件不存在", {}
        
        # 检查是否有有效标签
        if not tags_list:
            logging.debug(f"跳过组织：{os.path.basename(image_path)} - 无有效标签")
            return False, None, "无有效标签", {}
        
        # 使用分类器进行两层分类
        first_level, second_level, class_info = classifier.classify(tags_list)
        target_folder = class_info['folder_path']
        
        # 检查文件夹路径是否有效
        if not target_folder or target_folder == 'uncategorized/uncategorized' or not all(c.isalnum() or c in '_-/' for c in target_folder.replace('uncategorized', '')):
            logging.debug(f"跳过组织：{os.path.basename(image_path)} - 无效的分类文件夹: {target_folder}")
            return False, None, f"无效的分类文件夹: {target_folder}", class_info
        
        filename = os.path.basename(image_path)
        
        # 确定源目录和目标目录
        if base_dir:
            # 使用指定的基准目录
            target_dir = os.path.join(base_dir, target_folder)
        else:
            # 兼容旧逻辑：在原目录下创建子文件夹
            source_dir = os.path.dirname(image_path)
            target_dir = os.path.join(source_dir, target_folder)
        
        dst_path = os.path.join(target_dir, filename)
        
        # 如果目标文件已存在，返回目标路径但标记为跳过
        if os.path.exists(dst_path):
            logging.debug(f"跳过组织：目标文件已存在 - {dst_path}")
            return False, dst_path, "目标文件已存在", class_info
        
        # 创建文件夹并移动文件
        os.makedirs(target_dir, exist_ok=True)
        import shutil
        shutil.move(image_path, dst_path)
        
        logging.info(f"成功组织：{filename} -> {target_folder}/ 分类: {class_info['reason']}")
        return True, dst_path, None, class_info
        
    except Exception as e:
        error_msg = f"组织失败: {str(e)}"
        logging.error(f"组织图片 {image_path} 时发生错误: {error_msg}")
        return False, None, error_msg, {}

def main():
    program_start_time = time.perf_counter()
    print(f"程序启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logging.info(f"程序启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    print_step(0, "初始化 CUDA / ONNX Runtime 运行环境")
    env_info = detect_cuda_environment()
    print(f"CUDA 相关检测结果: {env_info}")




    # ！！！！！！！！！！！！！！！！！！！！需要手动改成正确的位置！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ 
    # 使用最新模型和对应的 JSON 标签映射
    latest_model_path = r"C:\danbooru-intel-onnx\最新模型\model_optimized.onnx"
    latest_tag_mapping_path = r"C:\danbooru-intel-onnx\最新模型\tag_mapping.json"
    labels_file_path = r"C:\danbooru-intel-onnx\tags.csv"
    label_column_name = 'tag'

    # ============================================================
    # 1. 运行环境与后端决策
    # ============================================================
    # 说明：
    # - 如果当前环境具备 CUDAExecutionProvider，优先使用 CUDA。
    # - 如果没有 CUDA，就退回到 CPUExecutionProvider。
    # - 同时保留原始模型候选列表，确保兼容历史路径。
    # ============================================================
    print_step(1, "检查 ONNX Runtime 和 CUDA 可用性")
    if ort is None:
        logging.error("onnxruntime 未安装，程序无法继续执行。")
        print("[ERROR] onnxruntime 未安装，请执行：pip install onnxruntime-gpu 或 onnxruntime")
        print_cuda_setup_guidance()
        input("按任意键退出...")
        return

    available_providers = ort.get_available_providers()
    print(f"ONNX Runtime 可用 providers: {available_providers}")
    logging.info(f"ONNX Runtime provider 列表: {available_providers}")

    prefer_cuda = "CUDAExecutionProvider" in available_providers
    if prefer_cuda:
        print("[OK] 检测到 CUDAExecutionProvider，后续将优先走 CUDA。")
        logging.info("CUDAExecutionProvider 可用，启用 CUDA 优先路径。")
    else:
        print("[WARN] 当前环境未检测到 CUDAExecutionProvider，后续将回退到 CPUExecutionProvider。")
        logging.warning("当前环境未检测到 CUDAExecutionProvider，使用 CPUExecutionProvider。")
        print_cuda_setup_guidance()

    # ============================================================
    # 2. 生成候选模型列表（CUDA 优先 / CPU 兜底）
    # ============================================================
    # 此处分成两段：
    # - CUDA 优先候选：更适合 NVIDIA 显卡
    # - CPU 兜底候选：在没有 CUDA 的环境中仍然尝试
    # ============================================================
    cuda_priority_candidates = [
        latest_model_path,
        r"C:\danbooru-intel-onnx\ml_caformer_m36_dec-5-97527.onnx",
        r"C:\Users\SNOW\Desktop\tagger_ml_danbooru\tagger_ml_danbooru\ml_danbooru.onnx",
        r"C:\sd-webui new\.cache\huggingface\models--deepghs--ml-danbooru-onnx\snapshots\60009d1a5989970203364a2b27c887e0fa2747f2\ml_caformer_m36_dec-5-97527.onnx",
        r"C:\sd-webui new\.cache\huggingface\models--SmilingWolf--wd-v1-4-moat-tagger-v2\snapshots\8452cddf280b952281b6e102411c50e981cb2908\model.onnx",
    ]
    cpu_fallback_candidates = [
        r"C:\Users\SNOW\Desktop\taggerV0.3\model.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\caformer_m36-3-80000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\ml_caformer_m36_dec-3-80000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\ml_caformer_m36_dec-5-97527.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_2-40000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_4-10000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_6-10000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_6-30000.onnx",
    ]

    MODEL_CANDIDATES = cuda_priority_candidates + cpu_fallback_candidates if prefer_cuda else cpu_fallback_candidates + cuda_priority_candidates
    print(f"候选模型顺序（按 backend 决策）: {MODEL_CANDIDATES}")
    logging.info(f"候选模型顺序: {MODEL_CANDIDATES}")

    COMMON_IMAGE_SIZES = [
        (448, 448),
        (480, 480), (512, 512), (640, 640), (768, 768), (800, 800), (1024, 1024)
    ]

    # =====================
    # 3. 加载标签文件
    # =====================
    labels = []
    if os.path.exists(latest_tag_mapping_path):
        logging.info(f"尝试从最新模型的 JSON 加载标签: {latest_tag_mapping_path}")
        labels = load_labels_from_json(latest_tag_mapping_path)

    if not labels:
        logging.info(f"JSON 标签加载失败，尝试从 CSV 加载: {labels_file_path}")
        labels = load_labels(labels_file_path, column_name=label_column_name)

    if not labels:
        logging.error("无法加载标签，程序退出。")
        print("错误：无法加载标签。请检查标签文件路径。")
        input("按任意键退出...")
        return

    logging.info(f"✓ 成功加载 {len(labels)} 个标签")
    print(f"成功加载标签数: {len(labels)}")

    # ========== 步骤 2：遍历模型并尝试加载 ==========
    found_working_model = False
    session = None
    final_model_path = None
    final_model_height = None
    final_model_width = None

    logging.info(f"开始枚举 {len(MODEL_CANDIDATES)} 个模型进行尝试。")
    print(f"开始枚举 {len(MODEL_CANDIDATES)} 个模型进行尝试...")

    for model_idx, current_model_path in enumerate(MODEL_CANDIDATES):
        logging.info(f"\n--- [{model_idx+1}/{len(MODEL_CANDIDATES)}] 尝试模型: {current_model_path} ---")
        print(f"\n--- [{model_idx+1}/{len(MODEL_CANDIDATES)}] 尝试模型: {os.path.basename(current_model_path)}")

        if not validate_model_file(current_model_path):
            continue

        for size_idx, (test_height, test_width) in enumerate(COMMON_IMAGE_SIZES):
            logging.info(f"      [{size_idx+1}/{len(COMMON_IMAGE_SIZES)}] 尝试输入尺寸: H={test_height}, W={test_width}")
            print(f"    尝试输入尺寸: H={test_height}, W={test_width}...")

            try:
                # ONNX Runtime 的 session 构建流程更直接，且允许选择 CUDA / CPU provider
                session = build_onnx_session(current_model_path, prefer_cuda=prefer_cuda)
                final_model_path = current_model_path
                final_model_height = test_height
                final_model_width = test_width
                found_working_model = True
                logging.info(f"成功加载模型: {current_model_path}, 输入尺寸: {(test_height, test_width)}")
                print(f"\n*** 成功找到可用模型: {os.path.basename(current_model_path)}, 目标尺寸: H={test_height}, W={test_width} ***")
                break
            except Exception as e:
                logging.warning(f"模型/尺寸尝试失败: {current_model_path}, size=({test_height},{test_width}), error={e}")
                print(f"[WARN] 模型尝试失败: {os.path.basename(current_model_path)}, size={test_height}x{test_width}, error={e}")

        if found_working_model:
            break

    model_load_complete_time = time.perf_counter()
    print(f"加载模型完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logging.info(f"加载模型完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    if not found_working_model or session is None:
        logging.error("无法加载或编译任何模型。")
        print("\n!!! 错误：所有尝试的模型和尺寸都无法加载或编译。")
        print_cuda_setup_guidance()
        input("按任意键退出...")
        return

    print_step(4, "模型加载完成，准备处理图片目录")
    print(f"最终使用模型: {final_model_path}")
    print(f"最终输入尺寸: {final_model_height}x{final_model_width}")
    print(f"Session providers: {session.get_providers()}")

    # ========== 加载配置文件并获取多个目录路径 ==========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'image_paths.txt')
    
    image_dirs = load_paths_from_file(config_file)
    
    if not image_dirs:
        logging.error("没有有效的目录路径")
        print("❌ 没有有效的目录路径")
        input("按任意键退出...")
        return
    
    # 获取用户确认
    if not get_user_confirmation(image_dirs):
        logging.info("用户取消操作")
        print("⏹ 已取消")
        input("按任意键退出...")
        return

    generate_txt_files = ask_yes_no("是否在对应图片文件夹中生成 txt 标签文件？输入 yes/no（默认 no）", default="no")
    print(f"[INFO] TXT 生成开关: {'启用' if generate_txt_files else '关闭'}")
    logging.info(f"TXT 生成开关: {'启用' if generate_txt_files else '关闭'}")
    
    user_threshold_str = input("请输入预测门限 (例如: 0.5, 范围 0.0-1.0): ").strip()
    try:
        user_threshold = float(user_threshold_str)
        if not (0.0 <= user_threshold <= 1.0):
            raise ValueError("门限值必须在 0.0 到 1.0 之间。")
        logging.info(f"用户输入的预测门限为: {user_threshold}")
    except:
        logging.error(f"无效的门限输入。将使用默认门限 0.2。")
        print(f"无效的门限输入。将使用默认门限 0.2。")
        user_threshold = 0.2

    # 初始化标签分类器，传入用户设定的阈值
    classifier = SimpleTagClassifier(debug=False, confidence_threshold=user_threshold)

    input_complete_time = time.perf_counter()
    print(f"输入完成开始运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logging.info(f"输入完成开始运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    # ========== 批量处理多个目录 ==========
    all_results_data = []
    all_organize_stats = {
        'organized': 0,
        'failed_organize': 0,
        'skipped_organize': 0
    }
    
    for dir_idx, input_folder in enumerate(image_dirs, 1):
        print(f"\n{'='*60}")
        print(f"[{dir_idx}/{len(image_dirs)}] 正在处理目录: {input_folder}")
        print(f"{'='*60}")
        logging.info(f"\n[{dir_idx}/{len(image_dirs)}] 正在处理目录: {input_folder}")
        
        # 验证目录
        if not os.path.isdir(input_folder):
            logging.error(f"输入的路径不是一个有效的文件夹：{input_folder}")
            print(f"⚠ 错误：'{input_folder}' 不是一个有效的文件夹路径，跳过。")
            continue

        image_files = []
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    image_files.append(os.path.abspath(os.path.join(root, file)))

        if not image_files:
            logging.info(f"在文件夹 {input_folder} 中没有找到支持的图片文件。")
            print(f"在文件夹 {input_folder} 中没有找到支持的图片文件。")
            continue

        logging.info(f"找到 {len(image_files)} 张图片进行推理。")
        results_data = []
        
        organize_stats = {
            'organized': 0,
            'failed_organize': 0,
            'skipped_organize': 0
        }

        retry_images = {}
        images_to_process_queue = list(image_files)
        total_images_in_folder = len(image_files)

        target_img_height = final_model_height
        target_img_width = final_model_width

        total_processing_time = 0
        total_preprocess_time = 0
        total_inference_time = 0
        total_postprocess_time = 0

        completed_images = set()
        
        # 创建总体进度条
        pbar_total = tqdm(total=total_images_in_folder, desc=f"目录 {dir_idx}/{len(image_dirs)}", unit="张", dynamic_ncols=True)

        # --- 批量处理循环 ---
        batch_count = 0
        while len(completed_images) < total_images_in_folder:
            current_batch_paths = []
            
            while len(current_batch_paths) < BATCH_SIZE and images_to_process_queue:
                img_path = images_to_process_queue.pop(0)
                if img_path not in completed_images:
                    # 检查文件名中是否包含 @@@，如果包含则跳过处理（已处理过）
                    filename = os.path.basename(img_path)
                    if '@@@' in filename:
                        logging.info(f"文件 {filename} 已处理（包含@@@标记），跳过处理")
                        results_data.append({
                            "文件名": filename,
                            "完整路径": img_path,
                            "预测标签": "已处理",
                            "置信度": "N/A",
                            "所有预测标签（含相似度）": "已处理过",
                            "所有预测标签": "已处理过",
                            "所有预测标签字数": 0,
                            "错误信息": "",
                            "重命名后的文件名": ""
                        })
                        completed_images.add(img_path)
                        pbar_total.update(1)
                        continue
                    current_batch_paths.append(img_path)
            
            if not images_to_process_queue and retry_images:
                for img_path_to_retry in list(retry_images.keys()):
                    if img_path_to_retry not in images_to_process_queue and img_path_to_retry not in completed_images:
                        images_to_process_queue.append(img_path_to_retry)
                while len(current_batch_paths) < BATCH_SIZE and images_to_process_queue:
                    img_path = images_to_process_queue.pop(0)
                    if img_path not in completed_images:
                        current_batch_paths.append(img_path)

            if not current_batch_paths:
                logging.info("当前批次为空，所有图片已处理或达到重试上限。")
                break

            batch_count += 1
            total_batches = (total_images_in_folder + BATCH_SIZE - 1) // BATCH_SIZE
            
            logging.info(f"正在处理批次 {batch_count}/{total_batches}，包含 {len(current_batch_paths)} 张图片。")

            batch_start_time = time.perf_counter()

            preprocess_start = time.perf_counter()
            batch_tensor, successful_flags_for_current_batch, preprocess_errors_for_current_batch = \
                preprocess_batch_images(current_batch_paths, target_img_height, target_img_width)
            preprocess_end = time.perf_counter()
            total_preprocess_time += (preprocess_end - preprocess_start)

            for i, img_path_in_batch in enumerate(current_batch_paths):
                if not successful_flags_for_current_batch[i]:
                    if img_path_in_batch not in completed_images:
                        results_data.append({
                            "文件名": os.path.basename(img_path_in_batch),
                            "完整路径": img_path_in_batch,
                            "预测标签": "预处理失败",
                            "置信度": 0.0,
                            "所有预测标签（含相似度）": "预处理失败",
                            "所有预测标签": "预处理失败",
                            "所有预测标签字数": 0,
                            "错误信息": f"预处理失败: {preprocess_errors_for_current_batch[i]}",
                            "重命名后的文件名": ""
                        })
                        completed_images.add(img_path_in_batch)
                        pbar_total.update(1)
                        logging.error(f"图片 {os.path.basename(img_path_in_batch)} 预处理失败。")

            if batch_tensor is None or batch_tensor.shape[0] == 0:
                logging.warning(f"批次中所有图片预处理失败，跳过推理。")
                continue

            try:
                logging.debug(f"推理前 batch_tensor 形状: {batch_tensor.shape}")
                
                if np.any(np.isnan(batch_tensor)) or np.any(np.isinf(batch_tensor)):
                    raise ValueError("推理输入包含 NaN 或 Inf 值。")

                inference_start = time.perf_counter()
                input_name = session.get_inputs()[0].name
                output_names = [output.name for output in session.get_outputs()]
                output_list = session.run(output_names, {input_name: batch_tensor})
                output_tensor = output_list[0]
                inference_end = time.perf_counter()
                total_inference_time += (inference_end - inference_start)

                logging.info(f"ONNX Runtime 推理完成。输入张量 shape={batch_tensor.shape}, 输出 shape={output_tensor.shape}, providers={session.get_providers()}")
                logging.debug(f"推理完成。output_tensor 形状: {output_tensor.shape}")

                postprocess_start = time.perf_counter()
                successful_original_paths = [path for i, path in enumerate(current_batch_paths) if successful_flags_for_current_batch[i]]
                
                with tqdm(total=len(successful_original_paths), desc=f"[目录{dir_idx}] 后处理", leave=False, unit="张") as pbar:
                    for j in range(output_tensor.shape[0]):
                        original_image_path = successful_original_paths[j]

                        if original_image_path in completed_images:
                            logging.debug(f"图片 {os.path.basename(original_image_path)} 已处理，跳过后处理。")
                            pbar.update(1)
                            continue

                        single_image_output = output_tensor[j]
                        predicted_label, confidence, all_predicted_tags_with_confidence_str, all_predicted_tags_pure_str, tags_list = \
                            postprocess_output(single_image_output, labels, threshold=user_threshold)

                        tag_words_count = sum(len(tag.strip()) for tag in all_predicted_tags_pure_str.split(',') if tag.strip())

                        result_entry = {
                            "文件名": os.path.basename(original_image_path),
                            "完整路径": original_image_path,
                            "预测标签": predicted_label,
                            "置信度": f"{confidence:.4f}",
                            "所有预测标签（含相似度）": all_predicted_tags_with_confidence_str,
                            "所有预测标签": all_predicted_tags_pure_str,
                            "所有预测标签字数": tag_words_count,
                            "错误信息": "",
                            "生成的TXT文件": ""
                        }
                        
                        # 生成TXT标签文件（可选，默认关闭，用户可选择 yes/no）
                        if generate_txt_files:
                            txt_success, txt_result, class_info = generate_txt_for_tags(original_image_path, tags_list, classifier)
                            if txt_success:
                                result_entry["生成的TXT文件"] = os.path.basename(txt_result)
                                organize_stats['organized'] += 1
                            elif txt_result in ["无有效标签", "文件已存在"]:
                                organize_stats['skipped_organize'] += 1
                            else:
                                organize_stats['failed_organize'] += 1
                        else:
                            result_entry["生成的TXT文件"] = "未生成"
                        
                        results_data.append(result_entry)
                        completed_images.add(original_image_path)
                        if original_image_path in retry_images:
                            del retry_images[original_image_path]

                        logging.info(f"图片 {os.path.basename(original_image_path)} 推理完成。标签: {predicted_label}, 置信度: {confidence:.4f}")
                        pbar_total.update(1)
                        pbar.update(1)
                        
                postprocess_end = time.perf_counter()
                total_postprocess_time += (postprocess_end - postprocess_start)

            except Exception as e:
                error_details = str(e)
                logging.error(f"批次推理/后处理异常：{error_details}")

                for k, img_path_in_batch in enumerate(current_batch_paths):
                    if img_path_in_batch in completed_images:
                        continue

                    if successful_flags_for_current_batch[k]:
                        current_retries = retry_images.get(img_path_in_batch, {"retry_count": 0, "last_error": ""})["retry_count"]
                        
                        if current_retries < MAX_RETRY_ATTEMPTS:
                            retry_images[img_path_in_batch] = {"retry_count": current_retries + 1, "last_error": error_details}
                            images_to_process_queue.append(img_path_in_batch)
                            print(f"图片 {os.path.basename(img_path_in_batch)} 预测失败，正在重试第 {current_retries + 1} 次...")
                            logging.warning(f"图片重试第 {current_retries + 1} 次。")
                        else:
                            results_data.append({
                                "文件名": os.path.basename(img_path_in_batch),
                                "完整路径": img_path_in_batch,
                                "预测标签": "预测失败",
                                "置信度": 0.0,
                                "所有预测标签（含相似度）": "预测失败",
                                "所有预测标签": "预测失败",
                                "所有预测标签字数": 0,
                                "错误信息": f"达到最大重试次数",
                                "重命名后的文件名": ""
                            })
                            completed_images.add(img_path_in_batch)
                            pbar_total.update(1)
                            if img_path_in_batch in retry_images:
                                del retry_images[img_path_in_batch]

            batch_end_time = time.perf_counter()
            total_processing_time += (batch_end_time - batch_start_time)

        # 关闭进度条
        pbar_total.close()

        # ========== 目录处理完成后的统计和收集 ==========
        all_results_data.extend(results_data)
        all_organize_stats['organized'] += organize_stats['organized']
        all_organize_stats['failed_organize'] += organize_stats['failed_organize']
        all_organize_stats['skipped_organize'] += organize_stats['skipped_organize']
        
        print(f"\n[{dir_idx}/{len(image_dirs)}] {input_folder} 处理完成")
        print(f"  本目录共处理: {len(image_files)} 张图片")
        print(f"  本目录TXT生成成功: {organize_stats['organized']}, 跳过: {organize_stats['skipped_organize']}, 失败: {organize_stats['failed_organize']}")

    # ========== 所有目录处理完成后的总体统计 ==========
    print(f"\n{'='*60}")
    print(f"✓ 所有目录处理完成！")
    print(f"{'='*60}")
    print("\n--- 总体统计信息 ---")
    print(f"处理的目录数: {len(image_dirs)}")
    print(f"总处理图片数量: {len(all_results_data)} 张")
    
    print("\n--- 整体TXT文件生成统计 ---")
    print(f"成功生成: {all_organize_stats['organized']}")
    print(f"跳过生成: {all_organize_stats['skipped_organize']}")
    print(f"生成失败: {all_organize_stats['failed_organize']}")

    try:
        csv_filename = f"image_tagging_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_csv_results(all_results_data, csv_filename)
        print(f"\n所有图片处理完成。默认 CSV 已保存到：{csv_filename}")

    except Exception as e:
        error_msg = f"保存默认 CSV 失败：{str(e)}"
        logging.error(error_msg)
        print(f"[ERROR] {error_msg}")

    finally:
        program_end_time = time.perf_counter()
        print(f"程序结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        logging.info(f"程序结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

        print("\n--- 阶段耗时统计 ---")
        print(f"从程序启动到加载模型完成: {model_load_complete_time - program_start_time:.4f} 秒")
        print(f"从加载模型完成到输入完成: {input_complete_time - model_load_complete_time:.4f} 秒")
        print(f"从输入完成到程序结束: {program_end_time - input_complete_time:.4f} 秒")

        try:
            if sys.platform == "win32":
                os.startfile(log_filename)
                logging.info(f"已自动打开日志文件。")
        except Exception as e:
            print(f"警告：无法自动打开日志文件：{e}")

    print("程序运行结束。")
    input("按任意键退出...")

if __name__ == "__main__":
    main()
