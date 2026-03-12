# image2pptx

`image2pptx` 会把一张扁平风格图片转换成一页 PowerPoint，并尽量把其中“适合编辑”的部分还原成可编辑对象。

当前项目更适合信息图、扁平插画、图标、卡片、箭头、流程图和 Gemini 风格的结构化图片。输出结果通常是混合形态：一部分是可编辑文本、一部分是 PowerPoint 原生图形、一部分是自由曲线，剩下较复杂的组件则会以透明 PNG 资产的形式放进 `PPTX`，优先保证视觉正确和元素可单独选中。

英文说明见 [`README.md`](README.md)。

## 当前流水线能产出什么

- OCR 识别出的可编辑文本框。
- 可编辑的 `rect`、`circle`、`line` 等基础图形。
- 一部分面积较大、形状较简单的可编辑自由曲线。
- 对复杂组件生成透明 PNG 资产，避免错误矢量化。
- 一套 sidecar artifacts，方便逐步排查每个阶段。

这个项目当前追求的是“实用重建”，不是“完美矢量还原”。目前的重点是：

- 让主要元素能被单独选中
- 尽量保留原图层级和颜色
- 避免生成几百上千个细碎对象导致幻灯片不可用

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

- `ultralytics` `SAM`
- `rapidocr-onnxruntime`
- `vtracer`

可选和平台相关依赖：

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

如果你有可用的 CUDA 环境，可以安装对应的 GPU 版本。

### 3. 安装项目依赖

```bash
python -m pip install -e .
```

如果你想在 Windows 上启用 PowerPoint COM 导出：

```bash
python -m pip install -e ".[windows]"
```

如果你还要跑本地测试：

```bash
python -m pip install -e ".[dev]"
```

## CLI 用法

最基本命令：

```bash
image2pptx input.png output.pptx
```

常见变体：

```bash
image2pptx input.png output.pptx --artifacts-dir artifacts
image2pptx input.png output.pptx --segmentation-backend cv
image2pptx input.png output.pptx --sam-model sam_b.pt
image2pptx input.png output.pptx --exporter python-pptx
image2pptx input.png output.pptx --exporter com
image2pptx input.png output.pptx --no-dump-scene-graph
```

当前 CLI 默认值：

- `--segmentation-backend sam`
- `--sam-model sam_b.pt`
- `--exporter auto`
- artifact 目录总会生成；如果不传 `--artifacts-dir`，默认会生成 `<output_stem>_artifacts`
- 默认会输出 `scene_graph.json`，除非传 `--no-dump-scene-graph`

第一次跑 `sam` 后端时，如果本地没有对应模型文件，Ultralytics 可能会自动下载所请求的权重。

## 流水线总览

下面这部分描述的是“当前代码真实实现”的 pipeline。

### 1. 解析输出路径和运行配置

CLI 入口是 `image2pptx`。程序会先校验 backend 和 exporter 参数，创建 `Image2PptxConfig`，再用 `--sam-model` 覆盖当前运行所用的 SAM 模型路径，并在正式处理前确定 artifact 目录位置。

### 2. 预处理输入图片

输入图片会先被统一读取成 `RGBA`。

当前预处理阶段会做这些事：

- 如果最长边超过 `1600px`，就按比例缩小
- 当 `blur_radius > 0` 时才做高斯模糊
- 默认把图片量化成 `8` 色的自适应调色板版本

这一阶段会输出：

- `preprocessed.png`
- `quantized.png`

`preprocessed.png` 是缩放后、可选模糊后的工作图。`quantized.png` 是降色后的版本，在 `cv` 后端中会直接参与后续分割，在 `sam` 后端中主要作为调试参考图保留。

### 3. OCR 和文字掩码

OCR 在预处理后的 `RGBA` 图上运行，当前使用的是 `rapidocr-onnxruntime`。

现在这一步的行为是：

- 只保留分数 `>= 0.5` 的文本框
- 给 OCR 框增加一定 padding
- 把检测结果转成可编辑文本节点
- 从局部图像内容估计文字颜色
- 根据 OCR 框高度估计字号

如果启用了文字掩码，流水线还会额外生成像素级 `text_mask`，避免后续把文字错误地并入图形组件。

这一阶段可能输出：

- `ocr_boxes.json`
- `text_mask.png`

如果运行环境里 OCR 不可用，或者没有识别到文本，流程会继续执行，只是不会生成文本节点。

### 4. 图形分割

当前项目支持两种分割后端。

### 4A. `sam` 后端

这是 CLI 现在的默认后端。

`sam` 路径使用的是预处理后的全彩图片，而不是 `quantized.png`。

在送入 SAM 之前：

- 如果前面得到了 `text_mask`，程序会先用 OpenCV inpainting 把文字区域“补掉”
- 补完文字的输入图会保存为 `sam_input.png`

随后通过 `ultralytics.SAM` 跑分割，再对原始 mask 做后处理，过滤掉这些情况：

- 面积太小
- 占整张图比例过大
- 在自身 bbox 内过于稀疏
- 和文本区域重叠太多
- 与已保留 mask 近似重复
- 被更优 mask 基本包含

最终保留的 mask 数量会被限制，并按面积排序。

`sam` 路径会输出：

- `sam_input.png`
- `sam_masks.json`
- `sam_masks_overlay.png`
- `component_boxes.png`
- `component_crops/`

### 4B. `cv` 后端

`cv` 后端直接基于量化后的 `RGBA` 图工作。

当前逻辑是：

1. 枚举所有非透明颜色
2. 为每种颜色生成二值 mask
3. 对每种颜色做连通域拆分
4. 提取轮廓
5. 把每个组件分类为 `rect`、`circle`、`line`、`freeform` 或 `svg_candidate`

这个后端的优势是可解释、好调试，但对 `quantized.png` 的质量非常敏感。

### 5. 把分割结果转成 scene graph

无论走哪条分割路径，最后都会构建一个 `SceneGraph`，再由导出器把它写成 PowerPoint。

当前 scene graph 里会出现的节点类型有：

- primitive shapes
- freeform polygons
- SVG asset nodes
- picture asset nodes
- text nodes

两条后端在这一步的主要差异是：

- `sam` 路径里，面积较大且形状简单的组件会变成可编辑图形；其余复杂组件会被裁成透明 PNG，写到 `component_assets/`
- `cv` 路径里，面积较大且形状简单的组件会保留为可编辑图形；其余区域会被绘制到 residual 图层，再拆成 `fallback_png/` 下的图片资产

当前代码在 `cv` 路径里也可能生成 `svg/` 中间文件，但最终 scene graph 仍然主要依赖 fallback 图像，而不是把这些 SVG 作为真正可编辑对象保留到最终幻灯片里。

文本节点会最后追加，所以通常会显示在图形之上。

如果启用了 scene graph 导出，这一阶段会写出：

- `scene_graph.json`

### 6. 导出为 `PPTX`

当前导出器选择逻辑是：

- `auto`：在 Windows 且安装了 `pywin32` 时优先尝试 COM，否则走 `python-pptx`
- `python-pptx`：始终使用纯 Python 导出
- `com`：强制使用 Windows PowerPoint COM 导出

当前导出行为包括：

- 幻灯片宽度固定为 `10in`
- 如果没有额外配置，高度会按原图宽高比自动推导
- 背景色来自 scene graph 中检测到的背景色
- 文本默认使用 `Microsoft YaHei`

`python-pptx` 会尽量插入原生 PowerPoint 图形，并对资产类节点使用图片回退。COM 导出器除了能插入 Office 原生图形外，在磁盘上存在 `.svg` 文件时也可以直接放入 SVG。

## Artifact 说明

artifact 目录总会生成。位置要么来自 `--artifacts-dir`，要么默认是 `<output_stem>_artifacts`。

通用产物：

- `preprocessed.png`：缩放后、可选模糊后的工作图
- `quantized.png`：降色图，`cv` 后端直接使用它做分割
- `scene_graph.json`：导出前的中间 scene graph，除非被显式关闭

OCR 相关产物：

- `ocr_boxes.json`：OCR 文字框和分数
- `text_mask.png`：保护文字区域的像素级掩码

`sam` 后端产物：

- `sam_input.png`：去掉文字后的 SAM 输入图
- `sam_masks.json`：原始与后处理后 mask 的元数据
- `sam_masks_overlay.png`：mask 叠加可视化图
- `component_boxes.png`：最终组件 bbox 可视化图
- `component_crops/`：每个最终组件的裁剪预览
- `component_assets/`：真正被 scene graph 引用的复杂组件透明 PNG

`cv` 后端产物：

- `fallback_png/`：没有保留为可编辑节点的 residual 图片资产
- `svg/`：可用时生成的中间矢量化输出

## 项目结构

```text
docs/                 设计说明和实现说明
examples/             示例素材和样例预期
src/image2pptx/       CLI、pipeline、预处理、OCR、分割、scene graph、导出器
tests/                回归测试
```

## 运行测试

```bash
python -m pytest
```

## 当前限制

- 当前更适合扁平信息图类输入，不适合照片和写实场景。
- 稠密文本，尤其是小字号或中英混排场景，OCR 仍可能不稳定。
- 很多复杂组件仍会以透明 PNG 资产形式进入 `PPTX`，而不是 PowerPoint 原生矢量对象。
- 细连接线和小装饰元素仍可能被切碎，或者被合并得不够理想。
- `cv` 后端对颜色量化结果非常敏感。
- `python-pptx` 路径对资产类节点仍主要依赖图片回退，而不是保留 SVG 的原生可编辑能力。
