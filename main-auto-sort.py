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


# --- 配置日志 ---
log_filename = f"openvino_inference_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
# 将日志级别设置为 DEBUG，以便捕获更多详细信息
logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

def setup_logging():
    """
    配置日志，确保在程序启动时生成新的日志文件。
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # 将日志级别设置为 DEBUG，以便捕获更多详细信息
    logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
    logging.info("日志系统初始化完成。")

setup_logging()

# --- 全局配置变量 ---
BATCH_SIZE = 32 # 批次大小
MODEL_EXPECTED_CHANNELS = 3
MAX_RETRY_ATTEMPTS = 3 # 最大重试次数

def preprocess_image_single(image_path, target_height, target_width):
    """
    预处理单张图片以符合模型输入要求。
    返回处理后的图片 NumPy 数组 (C, H, W)
    """
    image = None # 初始化image为None
    pil_read_success = False # 标志PIL是否成功读取
    error_message = "" # 记录预处理错误信息

    try:
        # --- 统一读取文件为字节流 ---
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            if not image_bytes:
                error_message = f"图片文件 {image_path} 读取为空。"
                logging.error(error_message)
                return None, error_message # 返回错误信息
            logging.debug(f"DEBUG: 成功读取 {image_path} 为字节流，长度: {len(image_bytes)}")
        except FileNotFoundError:
            error_message = f"图片文件 {image_path} 未找到。"
            logging.error(error_message)
            return None, error_message
        except Exception as e:
            error_message = f"读取图片文件 {image_path} 失败：{e}"
            logging.error(error_message)
            return None, error_message

        # --- 尝试使用OpenCV解码 ---
        raw_image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(raw_image_array, cv2.IMREAD_COLOR) # OpenCV默认读取BGR

        if image is None:
            logging.warning(f"图片 {image_path} OpenCV解码失败。尝试使用PIL。")
            # --- 如果OpenCV解码失败，尝试使用PIL ---
            try:
                # PIL直接从字节流读取
                pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                image = np.array(pil_img) # PIL读取的是RGB，转换为numpy数组
                pil_read_success = True
                logging.debug(f"DEBUG: {image_path} 通过PIL成功读取。形状: {image.shape}, 类型: {image.dtype}")
            except Exception as pil_e:
                error_message = f"图片 {image_path} PIL读取失败：{pil_e}"
                logging.error(error_message)
                image = None # 确保image仍然是None

        if image is None:
            if not error_message:
                error_message = f"无法读取图片：{image_path} (PIL和OpenCV均无法读取)"
            logging.error(error_message)
            return None, error_message # 返回错误信息

        # --- 统一处理图像数据 ---
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

        # 调整大小和归一化
        original_height, original_width = image.shape[0], image.shape[1]
        logging.debug(f"DEBUG: {image_path} 原始尺寸: H={original_height}, W={original_width}. 目标尺寸: H={target_height}, W={target_width}.")

        image = cv2.resize(image, (target_width, target_height))
        image = image.astype(np.float32) / 255.0
        image = image.transpose((2, 0, 1)) # C, H, W
        logging.debug(f"DEBUG: {image_path} 预处理完成。最终形状: {image.shape}, 数据范围: [{np.min(image):.4f}, {np.max(image):.4f}]")

        return image, None
    except Exception as e:
        error_message = f"图片预处理失败：{image_path}，错误：{e}"
        logging.error(error_message)
        return None, error_message

def preprocess_batch_images(image_paths, target_height, target_width):
    """
    预处理一个批次的图片。
    接收图片路径列表，返回一个批处理的 NumPy 数组 (N, C, H, W)。
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
    后处理模型输出，获取预测标签和置信度。此函数仍处理单张图片的输出。
    """
    try:
        scores = 1 / (1 + np.exp(-output_data_for_single_image))

        candidate_tags_with_scores = []
        for idx, score in enumerate(scores):
            if score >= threshold:
                if idx < len(labels):
                    candidate_tags_with_scores.append((labels[idx], score))
                else:
                    logging.warning(f"标签索引 {idx} 超出 labels 列表范围 ({len(labels)})。该高置信度标签将被忽略。")

        candidate_tags_with_scores.sort(key=operator.itemgetter(1), reverse=True)

        filtered_labels = [item[0] for item in candidate_tags_with_scores]
        filtered_scores = [item[1] for item in candidate_tags_with_scores]

        all_predicted_tags_with_confidence_str = ", ".join([f"{label} ({score:.4f})" for label, score in zip(filtered_labels, filtered_scores)])
        all_predicted_tags_pure_str = ", ".join(filtered_labels)

        predicted_label = filtered_labels[0] if filtered_labels else "无有效标签"
        confidence = filtered_scores[0] if filtered_scores else 0.0

        return predicted_label, confidence, all_predicted_tags_with_confidence_str, all_predicted_tags_pure_str

    except Exception as e:
        logging.error(f"后处理失败，错误：{e}")
        return "未知标签", 0.0, "后处理失败", "后处理失败"

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
        logging.error(f"标签文件未找到: {labels_file_path}。请确保该文件存在。")
        return []
    except Exception as e:
        logging.error(f"加载标签文件时发生错误：{e}")
        return []

def organize_images_immediately(image_path, predicted_label):
    """
    推理完成后立即整理图片到对应的标签文件夹
    参数：
        image_path: 图片的完整路径
        predicted_label: 预测的标签
    返回：
        (成功标志, 错误信息)
    """
    try:
        # 跳过失败的预测
        if '失败' in predicted_label or predicted_label == '无有效标签' or not predicted_label.strip():
            logging.debug(f"跳过组织：{os.path.basename(image_path)} - 无有效标签")
            return False, "无有效标签"
        
        # 检查源文件是否存在
        if not os.path.exists(image_path):
            logging.warning(f"组织失败：文件不存在 - {image_path}")
            return False, "文件不存在"
        
        filename = os.path.basename(image_path)
        source_dir = os.path.dirname(image_path)
        target_dir = os.path.join(source_dir, predicted_label)
        dst_path = os.path.join(target_dir, filename)
        
        # 如果目标文件已存在，跳过
        if os.path.exists(dst_path):
            logging.debug(f"跳过组织：目标文件已存在 - {dst_path}")
            return False, "目标文件已存在"
        
        # 创建文件夹并移动文件
        os.makedirs(target_dir, exist_ok=True)
        import shutil
        shutil.move(image_path, dst_path)
        
        logging.info(f"成功组织：{filename} -> {predicted_label}/")
        return True, None
        
    except Exception as e:
        error_msg = f"组织失败: {str(e)}"
        logging.error(f"组织图片 {image_path} 时发生错误: {error_msg}")
        return False, error_msg

def main():
    # --- 记录程序开始运行时间 ---
    program_start_time = time.perf_counter()
    print(f"程序启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logging.info(f"程序启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    # --- 标签文件路径 ---
    labels_file_path = r"C:\danbooru-intel-onnx\tags.csv"
    label_column_name = 'tag'

    # --- ONNX 模型文件路径列表 ---
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

    # --- 常见图片尺寸列表 (Height, Width) ---
    COMMON_IMAGE_SIZES = [
        (224, 224), (256, 256), (300, 300), (384, 384), (448, 448),
        (480, 480), (512, 512), (640, 640), (768, 768), (800, 800), (1024, 1024)
    ]

    # --- 加载标签 ---
    labels = load_labels(labels_file_path, column_name=label_column_name)
    if not labels:
        logging.error("无法加载标签，程序退出。")
        print("错误：无法加载标签。请检查标签文件路径和列名。")
        input("按任意键退出...")
        return

    # --- 初始化 OpenVINO 核心 ---
    try:
        core = ov.Core()
        logging.info("OpenVINO Core 初始化成功。")
    except Exception as e:
        logging.error(f"OpenVINO Core 初始化失败：{e}")
        print(f"错误：OpenVINO Core 初始化失败：{e}")
        input("按任意键退出...")
        return

    # --- 尝试加载并编译模型 ---
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
            print(f"模型文件不存在，跳过: {os.path.basename(current_model_path)}")
            continue

        for size_idx, (test_height, test_width) in enumerate(COMMON_IMAGE_SIZES):
            logging.info(f"      [{size_idx+1}/{len(COMMON_IMAGE_SIZES)}] 尝试编译尺寸: H={test_height}, W={test_width}")
            print(f"    尝试编译尺寸: H={test_height}, W={test_width}...")

            try:
                model = core.read_model(current_model_path)
                logging.info(f"模型 {os.path.basename(current_model_path)} 重新读取成功用于尺寸 {test_height}x{test_width} 尝试。")

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
                    logging.warning(f"设备 {device} 不可用，回退到 CPU。可用设备：{available_devices}")
                    device = "CPU"

                compile_config = {
                    "INFERENCE_PRECISION_HINT": "f32",
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    ov.properties.enable_profiling(): True
                }

                compiled_model = core.compile_model(model_with_preprocess, device_name=device,
                                                    config=compile_config)

                logging.info(f"成功编译模型: {os.path.basename(current_model_path)}, 尺寸: H={test_height}, W={test_width}, Batch Size: {BATCH_SIZE}, 设备: {device}。")
                print(f"\n*** 成功找到工作模型: {os.path.basename(current_model_path)}, 兼容尺寸: H={test_height}, W={test_width}, Batch Size: {BATCH_SIZE} ***")
                found_working_model = True
                final_model_path = current_model_path
                final_model_height = test_height
                final_model_width = test_width
                break
            except Exception as e:
                logging.warning(f"尝试模型 {os.path.basename(current_model_path)}, 尺寸 H={test_height}, W={test_width}, Batch Size {BATCH_SIZE} 失败，错误：{e}")
                print(f"    尝试失败。")

        if found_working_model:
            break

    model_load_complete_time = time.perf_counter()
    print(f"加载模型完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logging.info(f"加载模型完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    if not found_working_model:
        logging.error("所有模型和所有枚举的尺寸都尝试失败。无法加载或编译任何模型。")
        print("\n!!! 错误：所有尝试的模型和尺寸都无法加载或编译。")
        input("按任意键退出...")
        return

    # --- 获取用户输入的图片文件夹路径 ---
    input_folder = input("请输入包含图片文件的文件夹路径：").strip()
    if not os.path.isdir(input_folder):
        logging.error(f"输入的路径不是一个有效的文件夹：{input_folder}")
        print(f"错误：'{input_folder}' 不是一个有效的文件夹路径。")
        input("按任意键退出...")
        return

    # --- 获取用户输入的预测门限 ---
    user_threshold_str = input("请输入预测门限 (例如: 0.5, 范围 0.0-1.0): ").strip()
    try:
        user_threshold = float(user_threshold_str)
        if not (0.0 <= user_threshold <= 1.0):
            raise ValueError("门限值必须在 0.0 到 1.0 之间。")
        logging.info(f"用户输入的预测门限为: {user_threshold}")
    except ValueError as ve:
        logging.error(f"无效的门限输入：{user_threshold_str}。错误：{ve}。将使用默认门限 0.2。")
        print(f"无效的门限输入。将使用默认门限 0.2。")
        user_threshold = 0.2
    except Exception as e:
        logging.error(f"获取门限输入时发生未知错误：{e}。将使用默认门限 0.2。")
        print(f"获取门限输入时发生未知错误。将使用默认门限 0.2。")
        user_threshold = 0.2

    # --- 是否启用自动整理 ---
    auto_organize_choice = input("推理完成后是否自动整理文件到标签文件夹？ (y/n): ").strip().lower()
    auto_organize = auto_organize_choice == 'y'

    input_complete_time = time.perf_counter()
    print(f"输入完成开始运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logging.info(f"输入完成开始运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    # --- 遍历文件夹并进行推理 ---
    image_files = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_files.append(os.path.abspath(os.path.join(root, file)))

    if not image_files:
        logging.info(f"在文件夹 {input_folder} 中没有找到支持的图片文件。")
        print(f"在文件夹 {input_folder} 中没有找到支持的图片文件。")
        input("按任意键退出...")
        return

    logging.info(f"找到 {len(image_files)} 张图片进行推理。")
    results_data = []
    
    # 组织统计
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
            logging.info("DEBUG: 当前批次为空，所有图片可能已处理或达到重试上限。退出批处理循环。")
            break

        print(f"\n正在处理批次 (已完成: {len(completed_images)} / 总数: {total_images_in_folder})，当前批次: {len(current_batch_paths)} 张图片。")
        logging.info(f"正在处理批次，包含 {len(current_batch_paths)} 张图片。")

        batch_start_time = time.perf_counter()

        # --- 计时：预处理 ---
        preprocess_start = time.perf_counter()
        batch_tensor, successful_flags_for_current_batch, preprocess_errors_for_current_batch = \
            preprocess_batch_images(current_batch_paths, target_img_height, target_img_width)
        preprocess_end = time.perf_counter()
        total_preprocess_time += (preprocess_end - preprocess_start)

        # 记录预处理失败的图片
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
                        "错误信息": f"预处理失败: {preprocess_errors_for_current_batch[i]}"
                    })
                    completed_images.add(img_path_in_batch)
                    logging.error(f"图片 {os.path.basename(img_path_in_batch)} 预处理失败，已记录。")

        if batch_tensor is None or batch_tensor.shape[0] == 0:
            logging.warning(f"DEBUG: 批次中所有图片预处理失败，跳过推理。")
            continue

        try:
            logging.debug(f"DEBUG: 推理前 batch_tensor 形状: {batch_tensor.shape}")
            
            if np.any(np.isnan(batch_tensor)) or np.any(np.isinf(batch_tensor)):
                raise ValueError("推理输入包含 NaN 或 Inf 值。")

            # --- 计时：推理 ---
            inference_start = time.perf_counter()
            output_tensor = compiled_model([batch_tensor])[compiled_model.output(0)]
            inference_end = time.perf_counter()
            total_inference_time += (inference_end - inference_start)
            
            logging.debug(f"DEBUG: 推理完成。output_tensor 形状: {output_tensor.shape}")

            # --- 计时：后处理 ---
            postprocess_start = time.perf_counter()
            successful_original_paths = [path for i, path in enumerate(current_batch_paths) if successful_flags_for_current_batch[i]]
            
            for j in range(output_tensor.shape[0]):
                original_image_path = successful_original_paths[j]

                if original_image_path in completed_images:
                    logging.debug(f"DEBUG: 图片 {os.path.basename(original_image_path)} 已处理，跳过后处理。")
                    continue

                single_image_output = output_tensor[j]
                predicted_label, confidence, all_predicted_tags_with_confidence_str, all_predicted_tags_pure_str = \
                    postprocess_output(single_image_output, labels, threshold=user_threshold)

                tag_words_count = sum(len(tag.strip()) for tag in all_predicted_tags_pure_str.split(',') if tag.strip())

                results_data.append({
                    "文件名": os.path.basename(original_image_path),
                    "完整路径": original_image_path,
                    "预测标签": predicted_label,
                    "置信度": f"{confidence:.4f}",
                    "所有预测标签（含相似度）": all_predicted_tags_with_confidence_str,
                    "所有预测标签": all_predicted_tags_pure_str,
                    "所有预测标签字数": tag_words_count,
                    "错误信息": ""
                })
                completed_images.add(original_image_path)
                if original_image_path in retry_images:
                    del retry_images[original_image_path]

                logging.info(f"图片 {os.path.basename(original_image_path)} 推理完成。标签: {predicted_label}, 置信度: {confidence:.4f}")
                
                # --- 立即整理图片 ---
                if auto_organize:
                    organize_success, organize_error = organize_images_immediately(original_image_path, predicted_label)
                    if organize_success:
                        organize_stats['organized'] += 1
                        print(f"  [整理] {os.path.basename(original_image_path)} -> {predicted_label}/")
                    elif organize_error == "无有效标签" or organize_error == "目标文件已存在":
                        organize_stats['skipped_organize'] += 1
                    else:
                        organize_stats['failed_organize'] += 1
                        print(f"  [整理失败] {os.path.basename(original_image_path)}: {organize_error}")
                        
            postprocess_end = time.perf_counter()
            total_postprocess_time += (postprocess_end - postprocess_start)

        except Exception as e:
            error_details = str(e)
            logging.error(f"批次推理/后处理过程中发生异常：{error_details}")

            for k, img_path_in_batch in enumerate(current_batch_paths):
                if img_path_in_batch in completed_images:
                    continue

                if successful_flags_for_current_batch[k]:
                    current_retries = retry_images.get(img_path_in_batch, {"retry_count": 0, "last_error": ""})["retry_count"]
                    
                    if current_retries < MAX_RETRY_ATTEMPTS:
                        retry_images[img_path_in_batch] = {"retry_count": current_retries + 1, "last_error": error_details}
                        images_to_process_queue.append(img_path_in_batch)
                        print(f"图片 {os.path.basename(img_path_in_batch)} 预测失败，正在重试第 {current_retries + 1} 次...")
                        logging.warning(f"图片 {os.path.basename(img_path_in_batch)} 预测失败，正在重试第 {current_retries + 1} 次。")
                    else:
                        results_data.append({
                            "文件名": os.path.basename(img_path_in_batch),
                            "完整路径": img_path_in_batch,
                            "预测标签": "预测失败 (达到最大重试次数)",
                            "置信度": 0.0,
                            "所有预测标签（含相似度）": "预测失败",
                            "所有预测标签": "预测失败",
                            "所有预测标签字数": 0,
                            "错误信息": f"达到最大重试次数({MAX_RETRY_ATTEMPTS})"
                        })
                        completed_images.add(img_path_in_batch)
                        if img_path_in_batch in retry_images:
                            del retry_images[img_path_in_batch]

        batch_end_time = time.perf_counter()
        total_processing_time += (batch_end_time - batch_start_time)

    # --- 最终检查是否有遗漏的图片 ---
    for original_img_path in image_files:
        if original_img_path not in completed_images:
            logging.critical(f"WARNING: 图片 {original_img_path} 未在处理循环中被标记为完成或失败。")
            results_data.append({
                "文件名": os.path.basename(original_img_path),
                "完整路径": original_img_path,
                "预测标签": "未知错误",
                "置信度": 0.0,
                "所有预测标签（含相似度）": "未知错误",
                "所有预测标签": "未知错误",
                "所有预测标签字数": 0,
                "错误信息": "图片未在处理流程中被完整处理（未知情况）"
            })

    print("\n--- 计时结果 ---")
    print(f"总共处理图片数量: {len(image_files)} 张")
    print(f"所有图片总用时: {total_processing_time:.2f} 秒")
    print(f"  其中预处理总用时: {total_preprocess_time:.2f} 秒 ({total_preprocess_time / total_processing_time * 100:.2f}%)" if total_processing_time > 0 else "")
    print(f"  其中推理总用时: {total_inference_time:.2f} 秒 ({total_inference_time / total_processing_time * 100:.2f}%)" if total_processing_time > 0 else "")
    print(f"  其中后处理总用时: {total_postprocess_time:.2f} 秒 ({total_postprocess_time / total_processing_time * 100:.2f}%)" if total_processing_time > 0 else "")

    if len(image_files) > 0:
        average_time_per_image = total_processing_time / len(image_files)
        print(f"平均每张图片总用时: {average_time_per_image:.4f} 秒")
        print(f"  平均每张图片预处理用时: {total_preprocess_time / len(image_files):.4f} 秒")
        print(f"  平均每张图片推理用时: {total_inference_time / len(image_files):.4f} 秒")
        print(f"  平均每张图片后处理用时: {total_postprocess_time / len(image_files):.4f} 秒")

    # --- 打印整理统计 ---
    if auto_organize:
        print("\n--- 文件整理统计 ---")
        print(f"成功整理: {organize_stats['organized']}")
        print(f"跳过整理: {organize_stats['skipped_organize']}")
        print(f"整理失败: {organize_stats['failed_organize']}")

    # --- 存储结果到 XLSX ---
    try:
        df = pd.DataFrame(results_data)
        excel_filename = f"image_tagging_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(excel_filename, index=False)
        logging.info(f"结果已保存到 {excel_filename}")
        print(f"\n所有图片处理完成。结果已保存到：{excel_filename}")

        # 尝试自动打开 XLSX 文件
        if sys.platform == "win32":
            try:
                os.startfile(excel_filename)
                logging.info(f"已自动打开结果文件: {excel_filename}")
            except Exception as e:
                print(f"警告：无法自动打开结果文件：{e}")
                logging.warning(f"无法自动打开结果文件：{e}")

    except Exception as e:
        logging.error(f"保存结果到 XLSX 失败：{e}")
        print(f"错误：保存结果到 XLSX 失败：{e}")

    finally:
        # --- 记录程序结束时间 ---
        program_end_time = time.perf_counter()
        print(f"程序结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        logging.info(f"程序结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

        print("\n--- 阶段耗时统计 ---")
        print(f"从程序启动到加载模型完成: {model_load_complete_time - program_start_time:.4f} 秒")
        print(f"从加载模型完成到输入完成: {input_complete_time - model_load_complete_time:.4f} 秒")
        print(f"从输入完成到程序结束: {program_end_time - input_complete_time:.4f} 秒")

        # 尝试自动打开日志文件
        try:
            if sys.platform == "win32":
                os.startfile(log_filename)
                logging.info(f"已自动打开日志文件: {log_filename}")
        except Exception as e:
            print(f"警告：无法自动打开日志文件：{e}")
            logging.warning(f"无法自动打开日志文件：{e}")

    print("程序运行结束。")
    input("按任意键退出...")

if __name__ == "__main__":
    main()
