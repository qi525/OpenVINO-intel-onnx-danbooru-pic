from openvino.preprocess import PrePostProcessor, ColorFormat
import openvino as ov
import cv2
import numpy as np
import pandas as pd
import logging
import datetime
import os
import sys
import operator
import time
from PIL import Image
import io
import shlex
from typing import List, Tuple, Set
import re


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


# --- 配置日志 ---
log_filename = f"openvino_inference_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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

# --- 全局配置变量 ---
BATCH_SIZE = 32
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

def load_labels(labels_file_path, column_name=None):
    """
    从 CSV 文件中加载标签列表。
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

    labels_file_path = r"C:\danbooru-intel-onnx\tags.csv"
    label_column_name = 'tag'

    MODEL_CANDIDATES = [
        r"C:\danbooru-intel-onnx\ml_caformer_m36_dec-5-97527.onnx",
        r"C:\Users\SNOW\Desktop\tagger_ml_danbooru\tagger_ml_danbooru\ml_danbooru.onnx",
        r"C:\sd-webui new\.cache\huggingface\models--deepghs--ml-danbooru-onnx\snapshots\60009d1a5989970203364a2b27c887e0fa2747f2\ml_caformer_m36_dec-5-97527.onnx",
        r"C:\sd-webui new\.cache\huggingface\models--SmilingWolf--wd-v1-4-moat-tagger-v2\snapshots\8452cddf280b952281b6e102411c50e981cb2908\model.onnx",
        r"C:\Users\SNOW\Desktop\taggerV0.3\model.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\caformer_m36-3-80000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\ml_caformer_m36_dec-3-80000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\ml_caformer_m36_dec-5-97527.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_2-40000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_4-10000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_6-10000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_6-30000.onnx"
    ]

    COMMON_IMAGE_SIZES = [
        (224, 224), (256, 256), (300, 300), (384, 384), (448, 448),
        (480, 480), (512, 512), (640, 640), (768, 768), (800, 800), (1024, 1024)
    ]

    labels = load_labels(labels_file_path, column_name=label_column_name)
    if not labels:
        logging.error("无法加载标签，程序退出。")
        print("错误：无法加载标签。请检查标签文件路径和列名。")
        input("按任意键退出...")
        return

    try:
        core = ov.Core()
        logging.info("OpenVINO Core 初始化成功。")
    except Exception as e:
        logging.error(f"OpenVINO Core 初始化失败：{e}")
        print(f"错误：OpenVINO Core 初始化失败：{e}")
        input("按任意键退出...")
        return

    found_working_model = False
    compiled_model = None
    final_model_path = None
    final_model_height = None
    final_model_width = None

    logging.info(f"开始枚举 {len(MODEL_CANDIDATES)} 个模型进行尝试。")
    print(f"开始枚举 {len(MODEL_CANDIDATES)} 个模型进行尝试...")

    for model_idx, current_model_path in enumerate(MODEL_CANDIDATES):
        logging.info(f"\n--- [{model_idx+1}/{len(MODEL_CANDIDATES)}] 尝试模型: {current_model_path} ---")
        print(f"\n--- [{model_idx+1}/{len(MODEL_CANDIDATES)}] 尝试模型: {os.path.basename(current_model_path)}")

        if not os.path.exists(current_model_path):
            logging.warning(f"模型文件不存在，跳过: {current_model_path}")
            print(f"模型文件不存在，跳过。")
            continue

        for size_idx, (test_height, test_width) in enumerate(COMMON_IMAGE_SIZES):
            logging.info(f"      [{size_idx+1}/{len(COMMON_IMAGE_SIZES)}] 尝试编译尺寸: H={test_height}, W={test_width}")
            print(f"    尝试编译尺寸: H={test_height}, W={test_width}...")

            try:
                model = core.read_model(current_model_path)
                logging.info(f"模型读取成功。")

                ppp = PrePostProcessor(model)

                fixed_input_shape_for_preprocess = [BATCH_SIZE, MODEL_EXPECTED_CHANNELS, test_height, test_width]
                ppp.input(0).tensor() \
                    .set_shape(fixed_input_shape_for_preprocess) \
                    .set_layout(ov.Layout("NCHW")) \
                    .set_element_type(ov.Type.f32) \
                    .set_color_format(ColorFormat.RGB)

                ppp.output(0).postprocess().convert_element_type(ov.Type.f32)

                model_with_preprocess = ppp.build()

                device = "GPU"
                available_devices = core.available_devices
                if device not in available_devices:
                    logging.warning(f"设备 {device} 不可用，回退到 CPU。")
                    device = "CPU"

                compile_config = {
                    "INFERENCE_PRECISION_HINT": "f32",
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    ov.properties.enable_profiling(): True
                }

                compiled_model = core.compile_model(model_with_preprocess, device_name=device,
                                                    config=compile_config)

                logging.info(f"成功编译模型。")
                print(f"\n*** 成功找到工作模型: {os.path.basename(current_model_path)}, 兼容尺寸: H={test_height}, W={test_width} ***")
                found_working_model = True
                final_model_path = current_model_path
                final_model_height = test_height
                final_model_width = test_width
                break
            except Exception as e:
                logging.warning(f"尝试失败。")

        if found_working_model:
            break

    model_load_complete_time = time.perf_counter()
    print(f"加载模型完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logging.info(f"加载模型完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    if not found_working_model:
        logging.error("无法加载或编译任何模型。")
        print("\n!!! 错误：所有尝试的模型和尺寸都无法加载或编译。")
        input("按任意键退出...")
        return

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

    auto_organize_choice = input("推理完成后是否自动整理文件到标签文件夹？ (y/n): ").strip().lower()
    auto_organize = auto_organize_choice == 'y'

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

        # --- 批量处理循环 ---
        while len(completed_images) < total_images_in_folder:
            current_batch_paths = []
            
            while len(current_batch_paths) < BATCH_SIZE and images_to_process_queue:
                img_path = images_to_process_queue.pop(0)
                if img_path not in completed_images:
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

            print(f"\n正在处理批次 (已完成: {len(completed_images)} / 总数: {total_images_in_folder})，当前批次: {len(current_batch_paths)} 张图片。")
            logging.info(f"正在处理批次，包含 {len(current_batch_paths)} 张图片。")

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
                            "移动后的路径": ""
                        })
                        completed_images.add(img_path_in_batch)
                        logging.error(f"图片 {os.path.basename(img_path_in_batch)} 预处理失败。")

            if batch_tensor is None or batch_tensor.shape[0] == 0:
                logging.warning(f"批次中所有图片预处理失败，跳过推理。")
                continue

            try:
                logging.debug(f"推理前 batch_tensor 形状: {batch_tensor.shape}")
                
                if np.any(np.isnan(batch_tensor)) or np.any(np.isinf(batch_tensor)):
                    raise ValueError("推理输入包含 NaN 或 Inf 值。")

                inference_start = time.perf_counter()
                output_tensor = compiled_model([batch_tensor])[compiled_model.output(0)]
                inference_end = time.perf_counter()
                total_inference_time += (inference_end - inference_start)
                
                logging.debug(f"推理完成。output_tensor 形状: {output_tensor.shape}")

                postprocess_start = time.perf_counter()
                successful_original_paths = [path for i, path in enumerate(current_batch_paths) if successful_flags_for_current_batch[i]]
                
                for j in range(output_tensor.shape[0]):
                    original_image_path = successful_original_paths[j]

                    if original_image_path in completed_images:
                        logging.debug(f"图片 {os.path.basename(original_image_path)} 已处理，跳过后处理。")
                        continue

                    single_image_output = output_tensor[j]
                    predicted_label, confidence, all_predicted_tags_with_confidence_str, all_predicted_tags_pure_str, tags_list = \
                        postprocess_output(single_image_output, labels, threshold=user_threshold)

                    tag_words_count = sum(len(tag.strip()) for tag in all_predicted_tags_pure_str.split(',') if tag.strip())

                    # 创建结果条目
                    result_entry = {
                        "文件名": os.path.basename(original_image_path),
                        "完整路径": original_image_path,
                        "预测标签": predicted_label,
                        "置信度": f"{confidence:.4f}",
                        "所有预测标签（含相似度）": all_predicted_tags_with_confidence_str,
                        "所有预测标签": all_predicted_tags_pure_str,
                        "所有预测标签字数": tag_words_count,
                        "错误信息": "",
                        "移动后的路径": ""
                    }
                    
                    # 尝试组织文件并记录目标路径
                    if auto_organize:
                        organize_success, target_path, organize_error, class_info = organize_images_immediately(original_image_path, tags_list, classifier, base_dir=input_folder)
                        if organize_success:
                            result_entry["移动后的路径"] = target_path
                            organize_stats['organized'] += 1
                            print(f"  [整理] {os.path.basename(original_image_path)} -> {class_info.get('folder_path', 'unknown')}/")
                        elif organize_error == "无有效标签" or organize_error == "目标文件已存在":
                            if target_path:  # 如果目标路径存在则记录
                                result_entry["移动后的路径"] = target_path
                            organize_stats['skipped_organize'] += 1
                        else:
                            organize_stats['failed_organize'] += 1
                            print(f"  [整理失败] {os.path.basename(original_image_path)}: {organize_error}")
                    
                    results_data.append(result_entry)
                    completed_images.add(original_image_path)
                    if original_image_path in retry_images:
                        del retry_images[original_image_path]

                    logging.info(f"图片 {os.path.basename(original_image_path)} 推理完成。标签: {predicted_label}, 置信度: {confidence:.4f}")
                        
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
                                "移动后的路径": ""
                            })
                            completed_images.add(img_path_in_batch)
                            if img_path_in_batch in retry_images:
                                del retry_images[img_path_in_batch]

            batch_end_time = time.perf_counter()
            total_processing_time += (batch_end_time - batch_start_time)

        # ========== 目录处理完成后的统计和收集 ==========
        all_results_data.extend(results_data)
        all_organize_stats['organized'] += organize_stats['organized']
        all_organize_stats['failed_organize'] += organize_stats['failed_organize']
        all_organize_stats['skipped_organize'] += organize_stats['skipped_organize']
        
        print(f"\n[{dir_idx}/{len(image_dirs)}] {input_folder} 处理完成")
        print(f"  本目录共处理: {len(image_files)} 张图片")
        print(f"  本目录整理成功: {organize_stats['organized']}, 跳过: {organize_stats['skipped_organize']}, 失败: {organize_stats['failed_organize']}")

    # ========== 所有目录处理完成后的总体统计 ==========
    print(f"\n{'='*60}")
    print(f"✓ 所有目录处理完成！")
    print(f"{'='*60}")
    print("\n--- 总体统计信息 ---")
    print(f"处理的目录数: {len(image_dirs)}")
    print(f"总处理图片数量: {len(all_results_data)} 张")
    
    if auto_organize:
        print("\n--- 整体文件整理统计 ---")
        print(f"成功整理: {all_organize_stats['organized']}")
        print(f"跳过整理: {all_organize_stats['skipped_organize']}")
        print(f"整理失败: {all_organize_stats['failed_organize']}")

    try:
        df = pd.DataFrame(all_results_data)
        excel_filename = f"image_tagging_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(excel_filename, index=False)
        logging.info(f"结果已保存到 {excel_filename}")
        print(f"\n所有图片处理完成。结果已保存到：{excel_filename}")

        if sys.platform == "win32":
            try:
                os.startfile(excel_filename)
                logging.info(f"已自动打开结果文件。")
            except Exception as e:
                print(f"警告：无法自动打开结果文件：{e}")

    except Exception as e:
        logging.error(f"保存结果到 XLSX 失败：{e}")
        print(f"错误：保存结果到 XLSX 失败：{e}")

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
