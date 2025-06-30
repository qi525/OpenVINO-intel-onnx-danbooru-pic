import openvino as ov
from openvino.preprocess import PrePostProcessor, ColorFormat
import numpy as np
import cv2
import os
import pandas as pd
import datetime
import logging
import sys # 用于自动打开文件
import operator # 用于排序
import time # 用于计时
from PIL import Image # 用于图片读取的备选方案
import io # 用于PIL读取内存中的图片数据
import shlex # 用于安全引用命令行参数


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
    原理：优先尝试OpenCV解码图片。如果OpenCV解码失败，则尝试使用PIL库进行解码。PIL在处理某些非标准图片格式或元数据时可能更具鲁棒性。
    实现过程：
    1. 尝试使用Python内置open()函数读取文件为字节流，并增加调试日志记录文件字节数。
    2. 将字节流传递给cv2.imdecode读取图片。
    3. 如果cv2.imdecode返回None，捕获此情况。
    4. 在捕获到cv2.imdecode失败后，尝试使用PIL.Image.open从字节流读取图片，并将其转换为NumPy数组。
    5. 统一处理：无论通过何种方式读取，都确保图片为RGB三通道，并进行resize和归一化。
    主要改动点：
    - 将OpenCV的图片读取方式从直接使用np.fromfile和cv2.imdecode(path)改为先open()读取字节，再cv2.imdecode(bytes)。
    - 增加大量DEBUG级别的日志，记录图片读取的每一步状态和遇到的问题。
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
        # 改动点：将字节流转换为numpy数组，再传递给imdecode
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
            if not error_message: # 如果前面没有明确的错误信息，补充一个
                error_message = f"无法读取图片：{image_path} (PIL和OpenCV均无法读取)"
            logging.error(error_message)
            return None, error_message # 返回错误信息

        # --- 统一处理图像数据 ---
        # 如果是OpenCV读取的（BGR），需要转换为RGB
        if not pil_read_success: # 只有当PIL没有成功读取时，才可能是OpenCV读取的BGR
            if len(image.shape) == 3 and image.shape[2] == 3: # 确保是三通道彩色图
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                logging.debug(f"DEBUG: {image_path} (OpenCV读取) 转换为RGB。")
            elif len(image.shape) == 2: # 灰度图转换为RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                logging.debug(f"DEBUG: {image_path} (OpenCV读取) 灰度图转换为RGB。")
            else:
                error_message = f"图片 {image_path} 具有不支持的通道数或形状 ({image.shape})，无法转换为RGB。"
                logging.error(error_message)
                return None, error_message # 返回错误信息
        # 如果是PIL读取的，它已经通过.convert('RGB')确保是RGB

        # 调整大小和归一化
        original_height, original_width = image.shape[0], image.shape[1]
        logging.debug(f"DEBUG: {image_path} 原始尺寸: H={original_height}, W={original_width}. 目标尺寸: H={target_height}, W={target_width}.")

        image = cv2.resize(image, (target_width, target_height))
        image = image.astype(np.float32) / 255.0
        image = image.transpose((2, 0, 1)) # C, H, W
        logging.debug(f"DEBUG: {image_path} 预处理完成。最终形状: {image.shape}, 数据范围: [{np.min(image):.4f}, {np.max(image):.4f}]")

        return image, None # 成功返回图片和None错误信息
    except Exception as e:
        error_message = f"图片预处理失败：{image_path}，错误：{e}"
        logging.error(error_message)
        return None, error_message # 返回None图片和错误信息

def preprocess_batch_images(image_paths, target_height, target_width):
    """
    预处理一个批次的图片。
    接收图片路径列表，返回一个批处理的 NumPy 数组 (N, C, H, W)。
    同时返回一个布尔列表，指示哪些图片成功预处理，以及一个列表记录每张图片的预处理错误信息。
    改动点：不再填充0，只对成功预处理的图片进行堆叠。
    """
    preprocessed_images_successful = [] # 只存储成功预处理的图片
    successful_flags_for_all_inputs = [] # 标记每个输入图片是否成功预处理
    preprocess_errors_for_all_inputs = [] # 记录每个输入图片的预处理错误

    for image_path in image_paths:
        single_image, error_msg = preprocess_image_single(image_path, target_height, target_width)
        if single_image is not None:
            preprocessed_images_successful.append(single_image)
            successful_flags_for_all_inputs.append(True)
            preprocess_errors_for_all_inputs.append(None) # 成功则没有错误
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
    可以指定从哪一列读取标签。
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

def main():
    # --- 记录程序开始运行时间 ---
    program_start_time = time.perf_counter()
    print(f"程序启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logging.info(f"程序启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    # --- 标签文件路径 ---
    labels_file_path = r"C:\Users\SNOW\Desktop\tagger_ml_danbooru\tagger_ml_danbooru\tags.csv"
    label_column_name = 'tag'

    # --- >>> 你所有的 ONNX 模型文件路径列表 <<< ---
    MODEL_CANDIDATES = [
        r"C:\Users\SNOW\Desktop\tagger_ml_danbooru\tagger_ml_danbooru\ml_danbooru.onnx",
        r"C:\sd-webui new\.cache\huggingface\models--deepghs--ml-danbooru-onnx\snapshots\60009d1a5989970203364a2b27c887e0fa2747f2\ml_caformer_m36_dec-5-97527.onnx",
        r"C:\sd-webui new\.cache\huggingface\models--SmilingWolf--wd-v1-4-moat-tagger-v2\snapshots\8452cddf280b952281b6e102411c50e981cb2908\model.onnx", # WD 1.4 Moat Tagger V2
        r"C:\Users\SNOW\Desktop\taggerV0.3\model.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\caformer_m36-3-80000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\ml_caformer_m36_dec-3-80000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\ml_caformer_m36_dec-5-97527.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_2-40000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_4-10000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_6-10000.onnx",
        r"C:\个人数据\pythonCode\OpenVINO\ml-danbooru-onnx\TResnet-D-FLq_ema_6-30000.onnx"
    ]
    # --- <<< END OF MODEL CANDIDATES CONFIGURATION >>> ---

    # --- 暴力枚举尝试的常见图片尺寸列表 (Height, Width) ---
    COMMON_IMAGE_SIZES = [
        (224, 224),
        (256, 256),
        (300, 300),
        (384, 384),
        (448, 448),
        (480, 480),
        (512, 512),
        (640, 640),
        (768, 768),
        (800, 800),
        (1024, 1024)
    ]

    # --- 加载标签 ---
    labels = load_labels(labels_file_path, column_name=label_column_name)
    if not labels:
        logging.error("无法加载标签，程序退出。请检查标签文件路径和列名。")
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

                # 设置批量输入形状
                # 这里设置的BATCH_SIZE表示编译时可以处理的最大批次，实际推理时可以传入更小的批次
                # IMPORTANT: OpenVINO通常支持动态批次，只要编译时N设为-1或足够大的数。
                # 但为了安全起见，这里仍使用我们期望的BATCH_SIZE作为最大可能值。
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

                # --- 关键改动点：确保 OpenVINO 性能配置为 THROUGHPUT ---
                compile_config = {
                    "INFERENCE_PRECISION_HINT": "f32",
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    # 新增：启用性能分析，OpenVINO可能会根据这个提示调整内部行为
                    ov.properties.enable_profiling(): True
                }
                # --- 改动结束 ---

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
                print(f"    尝试失败。错误：{e}")
                logging.warning(f"    发生错误：{e}。继续尝试下一个尺寸或模型。")

        if found_working_model:
            break

    # --- 记录加载模型完成时间 ---
    model_load_complete_time = time.perf_counter()
    print(f"加载模型完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logging.info(f"加载模型完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    if not found_working_model:
        logging.error("所有模型和所有枚举的尺寸都尝试失败。无法加载或编译任何模型。")
        print("\n!!! 错误：所有尝试的模型和尺寸都无法加载或编译。")
        print("请检查：")
        print("1. 你的 ONNX 模型文件是否损坏。")
        print("2. 'COMMON_IMAGE_SIZES' 列表中是否包含了你的模型可能使用的**训练或推荐尺寸**。")
        print("3. OpenVINO 版本是否与模型兼容。")
        print(f"4. 尝试降低 Batch Size (当前设置为 {BATCH_SIZE})。")
        print("5. 如果问题持续，请尝试使用 OpenVINO Model Optimizer 将 ONNX 模型转换为 IR 格式，看是否能获得更具体的错误。")
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
        print(f"无效的门限输入：'{user_threshold_str}'。错误：{ve}。将使用默认门限 0.2。")
        user_threshold = 0.2 # 错误时使用默认值
    except Exception as e:
        logging.error(f"获取门限输入时发生未知错误：{e}。将使用默认门限 0.2。")
        print(f"获取门限输入时发生未知错误：{e}。将使用默认门限 0.2。")
        user_threshold = 0.2 # 错误时使用默认值

    # --- 记录输入完成时间 ---
    input_complete_time = time.perf_counter()
    print(f"输入完成开始运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logging.info(f"输入完成开始运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")


    # --- 遍历文件夹并进行推理 ---
    image_files = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(supported_extensions):
                # 确保完整路径是规范化的绝对路径
                image_files.append(os.path.abspath(os.path.join(root, file)))

    if not image_files:
        logging.info(f"在文件夹 {input_folder} 中没有找到支持的图片文件。")
        print(f"在文件夹 {input_folder} 中没有找到支持的图片文件。")
        input("按任意键退出...")
        return

    logging.info(f"找到 {len(image_files)} 张图片进行推理。")
    results_data = [] # 存储最终结果

    # 存储重试图片及其最后一次的错误信息
    # {image_path: {"retry_count": count, "last_error": "message"}}
    retry_images = {}
    
    # images_to_process 初始包含所有图片文件
    images_to_process_queue = list(image_files) # 使用队列来管理待处理图片，支持 pop(0)
    
    total_images_in_folder = len(image_files) # 记录文件夹中总图片数，用于进度显示

    target_img_height = final_model_height
    target_img_width = final_model_width

    total_processing_time = 0
    total_preprocess_time = 0
    total_inference_time = 0
    total_postprocess_time = 0

    # 用于跟踪已完成处理的图片，避免重复添加到results_data
    # 集合中存储图片的完整路径，用于快速查找
    completed_images = set()

    # --- 批量处理循环 (包含重试逻辑) ---
    while len(completed_images) < total_images_in_folder:
        current_batch_paths = []
        
        # 优先从 images_to_process_queue 中取图片
        # 只要队列中有图片，就尝试填充批次
        while len(current_batch_paths) < BATCH_SIZE and images_to_process_queue:
            img_path = images_to_process_queue.pop(0)
            if img_path not in completed_images: # 确保不处理已完成的图片
                current_batch_paths.append(img_path)
        
        # 如果 images_to_process_queue 已经空了，但还有图片需要重试，将其重新加入队列
        # 这确保了即使是重试失败的图片，也能被考虑在内，直到达到最大重试次数
        if not images_to_process_queue and retry_images:
            for img_path_to_retry in list(retry_images.keys()):
                if img_path_to_retry not in images_to_process_queue and img_path_to_retry not in completed_images:
                    images_to_process_queue.append(img_path_to_retry)
            # 重新尝试从 images_to_process_queue 中取图片，填充批次
            while len(current_batch_paths) < BATCH_SIZE and images_to_process_queue:
                img_path = images_to_process_queue.pop(0)
                if img_path not in completed_images:
                    current_batch_paths.append(img_path)

        if not current_batch_paths: # 如果当前批次没有图片，说明所有图片都已处理或无法处理
            logging.info("DEBUG: 当前批次为空，所有图片可能已处理或达到重试上限。退出批处理循环。")
            break

        print(f"\n正在处理批次 (已完成: {len(completed_images)} / 总数: {total_images_in_folder})，当前批次: {len(current_batch_paths)} 张图片。")
        logging.info(f"正在处理批次，包含 {len(current_batch_paths)} 张图片。")

        batch_start_time = time.perf_counter() # 批次总开始时间

        # --- 计时：预处理 ---
        preprocess_start = time.perf_counter()
        # preprocess_batch_images 现在只对成功预处理的图片堆叠为张量
        batch_tensor, successful_flags_for_current_batch, preprocess_errors_for_current_batch = \
            preprocess_batch_images(current_batch_paths, target_img_height, target_img_width)
        preprocess_end = time.perf_counter()
        total_preprocess_time += (preprocess_end - preprocess_start)

        # 记录预处理失败的图片到 results_data
        for i, img_path_in_batch in enumerate(current_batch_paths):
            if not successful_flags_for_current_batch[i]:
                if img_path_in_batch not in completed_images: # 避免重复添加
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
                    logging.error(f"图片 {os.path.basename(img_path_in_batch)} 预处理失败，已记录。绝对路径：{img_path_in_batch}")

        # 如果批次中没有成功预处理的图片，则跳过推理
        if batch_tensor is None or batch_tensor.shape[0] == 0:
            logging.warning(f"DEBUG: 批次 {current_batch_paths} 中所有图片预处理失败，跳过推理。")
            continue # 进入下一个批次循环

        try:
            # --- 调试信息：推理前检查batch_tensor ---
            logging.debug(f"DEBUG: 推理前 batch_tensor 形状: {batch_tensor.shape}")
            logging.debug(f"DEBUG: 推理前 batch_tensor 最小值: {np.min(batch_tensor):.4f}, 最大值: {np.max(batch_tensor):.4f}")
            logging.debug(f"DEBUG: 推理前 batch_tensor 是否包含NaN: {np.any(np.isnan(batch_tensor))}")
            logging.debug(f"DEBUG: 推理前 batch_tensor 是否包含Inf: {np.any(np.isinf(batch_tensor))}")
            if np.any(np.isnan(batch_tensor)) or np.any(np.isinf(batch_tensor)):
                raise ValueError("推理输入包含 NaN 或 Inf 值，可能导致推理失败。")

            # --- 计时：推理 ---
            inference_start = time.perf_counter()
            # OpenVINO会自动处理动态批次大小，只要它不超过编译时设定的最大批次
            output_tensor = compiled_model([batch_tensor])[compiled_model.output(0)]
            inference_end = time.perf_counter()
            total_inference_time += (inference_end - inference_start)
            logging.debug(f"DEBUG: 推理完成。output_tensor 形状: {output_tensor.shape}")
            logging.debug(f"DEBUG: 推理完成。output_tensor 最小值: {np.min(output_tensor):.4f}, 最大值: {np.max(output_tensor):.4f}")
            logging.debug(f"DEBUG: 推理完成。output_tensor 是否包含NaN: {np.any(np.isnan(output_tensor))}")
            logging.debug(f"DEBUG: 推理完成。output_tensor 是否包含Inf: {np.any(np.isinf(output_tensor))}")
            if np.any(np.isnan(output_tensor)) or np.any(np.isinf(output_tensor)):
                raise ValueError("模型输出包含 NaN 或 Inf 值，可能表示推理过程中出现问题。")

            # --- 计时：后处理 ---
            postprocess_start = time.perf_counter()
            # 建立成功预处理图片与 output_tensor 索引的映射
            successful_original_paths = [path for i, path in enumerate(current_batch_paths) if successful_flags_for_current_batch[i]]
            
            for j in range(output_tensor.shape[0]): # output_tensor.shape[0] 是实际成功推理的图片数量
                original_image_path = successful_original_paths[j]

                if original_image_path in completed_images: # 再次检查，避免在重试后被重复处理
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
            postprocess_end = time.perf_counter()
            total_postprocess_time += (postprocess_end - postprocess_start)

        except Exception as e:
            error_details = str(e)
            logging.error(f"批次推理/后处理过程中发生异常：{error_details}")

            # 遍历当前批次中的所有图片，进行重试或标记失败
            for k, img_path_in_batch in enumerate(current_batch_paths):
                if img_path_in_batch in completed_images:
                    continue # 已经处理过的图片跳过

                # 只有预处理成功的图片才考虑重试
                if successful_flags_for_current_batch[k]:
                    current_retries = retry_images.get(img_path_in_batch, {"retry_count": 0, "last_error": ""})["retry_count"]
                    
                    if current_retries < MAX_RETRY_ATTEMPTS:
                        retry_images[img_path_in_batch] = {"retry_count": current_retries + 1, "last_error": error_details}
                        images_to_process_queue.append(img_path_in_batch) # 重新加入待处理队列
                        print(f"图片 {os.path.basename(img_path_in_batch)} 预测失败，正在重试第 {current_retries + 1} 次... (错误: {error_details})")
                        logging.warning(f"图片 {os.path.basename(img_path_in_batch)} 预测失败，正在重试第 {current_retries + 1} 次。绝对路径：{img_path_in_batch}。错误：{error_details}")
                    else:
                        results_data.append({
                            "文件名": os.path.basename(img_path_in_batch),
                            "完整路径": img_path_in_batch,
                            "预测标签": "预测失败 (达到最大重试次数)",
                            "置信度": 0.0,
                            "所有预测标签（含相似度）": "预测失败",
                            "所有预测标签": "预测失败",
                            "所有预测标签字数": 0,
                            "错误信息": f"达到最大重试次数({MAX_RETRY_ATTEMPTS})，最后错误: {retry_images[img_path_in_batch]['last_error'] if img_path_in_batch in retry_images else error_details}"
                        })
                        completed_images.add(img_path_in_batch)
                        if img_path_in_batch in retry_images:
                            del retry_images[img_path_in_batch]

                        print(f"图片 {os.path.basename(img_path_in_batch)} 预测失败，已达到最大重试次数 {MAX_RETRY_ATTEMPTS}，报告失败。绝对路径：{img_path_in_batch}")
                        logging.error(f"图片 {os.path.basename(img_path_in_batch)} 预测失败，已达到最大重试次数 {MAX_RETRY_ATTEMPTS}，报告失败。绝对路径：{img_path_in_batch}")
                # 预处理失败的图片已经在前面记录，这里不重复处理

        batch_end_time = time.perf_counter()
        total_processing_time += (batch_end_time - batch_start_time)
        
    # --- 批量处理循环结束 ---

    # 最终检查是否有遗漏的图片（理论上不应该有，但以防万一）
    for original_img_path in image_files:
        if original_img_path not in completed_images:
            # 这表示一张图片既没有成功处理，也没有被记录为预处理失败或推理失败并达到重试上限
            # 这通常不应该发生，但如果发生，需要记录为未知错误
            logging.critical(f"WARNING: 图片 {original_img_path} 未在处理循环中被标记为完成或失败。将其标记为未知错误。")
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
    print(f"  其中预处理总用时: {total_preprocess_time:.2f} 秒 ({total_preprocess_time / total_processing_time * 100:.2f}%)")
    print(f"  其中推理总用时: {total_inference_time:.2f} 秒 ({total_inference_time / total_processing_time * 100:.2f}%)")
    print(f"  其中后处理总用时: {total_postprocess_time:.2f} 秒 ({total_postprocess_time / total_processing_time * 100:.2f}%)")


    if len(image_files) > 0:
        average_time_per_image = total_processing_time / len(image_files)
        print(f"平均每张图片总用时: {average_time_per_image:.4f} 秒")
        print(f"  平均每张图片预处理用时: {total_preprocess_time / len(image_files):.4f} 秒")
        print(f"  平均每张图片推理用时: {total_inference_time / len(image_files):.4f} 秒")
        print(f"  平均每张图片后处理用时: {total_postprocess_time / len(image_files):.4f} 秒")
    else:
        print("未处理任何图片，无法计算平均用时。")

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
                logging.info(f"已自动打开结果文件 (Windows): {excel_filename}")
            except Exception as e:
                print(f"警告：无法自动打开结果文件 {excel_filename} (Windows)：{e}")
                logging.warning(f"无法自动打开结果文件 {excel_filename} (Windows)：{e}")
        else: # macOS 或 Linux
            try:
                # 使用 shlex.quote 安全地引用文件名
                quoted_excel_filename = shlex.quote(excel_filename)
                os.system(f"xdg-open {quoted_excel_filename} 2>/dev/null || open {quoted_excel_filename}")
                logging.info(f"已自动打开结果文件 (Unix/Linux/macOS): {excel_filename}")
            except Exception as e:
                print(f"警告：无法自动打开结果文件 {excel_filename} (Unix/Linux/macOS)：{e}")
                logging.warning(f"无法自动打开结果文件 {excel_filename} (Unix/Linux/macOS)：{e}")
        

    except Exception as e:
        logging.error(f"保存结果到 XLSX 或自动打开文件失败：{e}")
        print(f"错误：保存结果到 XLSX 或自动打开文件失败：{e}")

    finally:
        # --- 记录程序结束时间 ---
        program_end_time = time.perf_counter()
        print(f"程序结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        logging.info(f"程序结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

        # --- 计算并打印各阶段耗时 ---
        print("\n--- 阶段耗时统计 ---")
        print(f"从程序启动到加载模型完成: {model_load_complete_time - program_start_time:.4f} 秒")
        print(f"从加载模型完成到输入完成: {input_complete_time - model_load_complete_time:.4f} 秒")
        print(f"从输入完成到程序结束: {program_end_time - input_complete_time:.4f} 秒")
        logging.info(f"从程序启动到加载模型完成: {model_load_complete_time - program_start_time:.4f} 秒")
        logging.info(f"从加载模型完成到输入完成: {input_complete_time - model_load_complete_time:.4f} 秒")
        logging.info(f"从输入完成到程序结束: {program_end_time - input_complete_time:.4f} 秒")

        # 尝试自动打开日志文件
        try:
            if sys.platform == "win32":
                os.startfile(log_filename)
                logging.info(f"已自动打开日志文件 (Windows): {log_filename}")
            else: # macOS 或 Linux
                # 使用 shlex.quote 安全地引用文件名
                quoted_log_filename = shlex.quote(log_filename)
                os.system(f"xdg-open {quoted_log_filename} 2>/dev/null || open {quoted_log_filename}")
                logging.info(f"已自动打开日志文件 (Unix/Linux/macOS): {log_filename}")
        except Exception as e:
            print(f"警告：无法自动打开日志文件 {log_filename}：{e}")
            logging.warning(f"无法自动打开日志文件 {log_filename}：{e}")

    print("程序运行结束。")
    input("按任意键退出...")

if __name__ == "__main__":
    main()