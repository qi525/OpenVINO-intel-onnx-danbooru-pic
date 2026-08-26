最新的逆推模型： 
https://huggingface.co/cella110n/cl_tagger/tree/main/cl_tagger_1_02


952行 代码需要手动改，模型及其配置文件的位置

684行 代码需要手动改，处理的目标文件夹




## 4. 安装依赖

```powershell
python -m pip install --upgrade pip
pip install numpy pandas opencv-python pillow tqdm
pip install onnx
```

如果电脑有 NVIDIA 显卡：

```powershell
pip install onnxruntime-gpu
```

如果没有 NVIDIA 显卡：

```powershell
pip install onnxruntime
```

---

## 5. 检查是否能导入

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

如果看到 `CPUExecutionProvider`，说明能跑。  
如果看到 `CUDAExecutionProvider`，说明支持 CUDA。  
如果报错，重新装依赖。

---

## 6. 把路径改对

打开脚本：

952行 代码需要手动改，模型及其配置文件的位置

```python
latest_model_path = r"C:\danbooru-intel-onnx\最新模型\model_optimized.onnx"
latest_tag_mapping_path = r"C:\danbooru-intel-onnx\最新模型\tag_mapping.json"
labels_file_path = r"C:\danbooru-intel-onnx\tags.csv"
```


---


684行 代码需要手动改，处理的目标文件夹

```python
        example_paths = [
            "# 在下方添加要处理的图片目录路径，每行一个",
            "# 例如:",
            r"C:\stable-diffusion-webui-reForge\outputs\txt2img-images\2026-03-28",
            r"D:\images\batch2",
            r"E:\Danbooru\sorted",
        ]
```



把它们改成你电脑里真实对应文件的路径。

另外也要确认 `image_paths.txt` 里写的是图片目录。

---
## 7. 运行脚本

```powershell
python main-auto-sort-v4-txt-for-cuda.py
```

---

## 8. 脚本会问什么

运行后，脚本一般会问：

- 是否确认处理这些目录
- 输入预测门限，例如 `0.5`
- 是否生成 txt 到图片同目录（yes/no）

---

## 9. 默认输出

默认会生成 CSV，表头是：

```csv
路径,tag
```

不生成 txt 的情况下，脚本只输出 CSV。

如果输入 `yes`，才会在图片同目录生成 `.txt` 文件。

---








