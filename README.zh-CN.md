# image2pptx

`image2pptx` 是一个使用 Python 实现的图像转 `PPTX` 项目，目标是把信息图、扁平插画、图标类图片尽量还原为可编辑或半可编辑的 PowerPoint 页面。

当前版本主要针对：

- 扁平风格插画
- 信息图面板
- 图标、箭头、数据库、卡片等组件
- Gemini 生成的结构化图片
- 需要保留调试产物、便于继续迭代的本地工作流

英文说明见 [`README.md`](README.md)。

## 项目当前做法

当前流水线不是简单把整张图塞进 `PPTX`，而是组合了几条链路：

1. `OCR` 识别文字并转为可编辑文本框。
2. `MobileSAM` 做组件级分割，尽量把数据库、脑图、卡片、图标、箭头拆成独立元素。
3. 用规则把简单组件识别为矩形、圆形、直线等 PowerPoint 原生对象。
4. 对暂时无法安全矢量化的复杂组件，先导出为透明图片资产，至少保证单独可选、位置正确、颜色接近原图。

当前阶段的目标是：

- 主要元素能独立选中
- 颜色尽量接近原图
- 避免生成几百上千个碎片形状导致 `PPTX` 卡死

## 技术栈

核心运行时：

- Python `3.11+`
- `opencv-python`
- `Pillow`
- `numpy`
- `python-pptx`
- `typer`
- `pydantic`

分割与 OCR：

- `ultralytics` + `MobileSAM`
- `rapidocr-onnxruntime`

可选依赖：

- `pywin32`
  用于 Windows 下通过 PowerPoint COM 导出
- `pytest`
  用于本地回归测试

## 本地部署

### 1. 创建并激活虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 安装 PyTorch

如果你在本地走 CPU：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

如果你有可用的 CUDA 环境，可以安装对应 GPU 版本。

### 3. 安装项目依赖

```bash
python -m pip install -e .
```

如果你希望在 Windows 上启用 PowerPoint COM 导出：

```bash
python -m pip install -e ".[windows]"
```

如果你还需要本地跑测试：

```bash
python -m pip install -e ".[dev]"
```

## 首次运行说明

第一次使用 `sam` 分割后端时，Ultralytics 可能会自动下载 `mobile_sam.pt` 权重文件。

最基本命令：

```bash
image2pptx input.png output.pptx
```

指定调试产物目录：

```bash
image2pptx input.png output.pptx --artifacts-dir artifacts
```

如果你想改回 OpenCV 规则分割：

```bash
image2pptx input.png output.pptx --segmentation-backend cv
```

如果你只想使用纯 `python-pptx` 导出：

```bash
image2pptx input.png output.pptx --exporter python-pptx
```

如果你在 Windows 上装了 PowerPoint 和 `pywin32`，可以尝试：

```bash
image2pptx input.png output.pptx --exporter com
```

## 调试产物

为了方便定位问题，项目会输出一组调试文件，例如：

- `preprocessed.png`
- `quantized.png`
- `text_mask.png`
- `sam_input.png`
- `ocr_boxes.json`
- `sam_masks.json`
- `sam_masks_overlay.png`
- `component_boxes.png`
- `component_crops/`
- `scene_graph.json`

这些文件可以帮助你判断问题出在：

- OCR 识别
- SAM 自动分割
- 掩码后处理
- 还是最终导出

## 项目结构

```text
docs/                 设计说明和指导手册
examples/             示例素材与样例说明
src/image2pptx/       CLI、流水线、导出器、OCR、SAM、SceneGraph 等核心代码
tests/                项目源码测试
```

## 运行测试

```bash
python -m pytest
```

## 当前限制

- 中文密集信息图的 OCR 质量仍然不够稳定。
- 很多复杂组件当前还是以“独立透明图片资产”形式进入 `PPTX`，还不是原生矢量形状。
- `MobileSAM` 已经能把主要组件拆开，但对细碎装饰、小图标内部结构、细连接线的处理还需要继续增强。
- 当前更适合信息图、扁平插画类图片，不适合照片或写实场景。
