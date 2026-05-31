# AI 图片关键词分类器

CLIP和SigLIP 模型零样本/自定样本的本地CUDA图片分类工具，按关键词分类/手动训练分类器，Powered by Claude&Python

## 功能

- **CLIP ViT-L-14** — 余弦相似度匹配，fp16 加速，batch 64
- **SigLIP ViT-L-14-384** — 概率分布匹配，pipeline API
- 普通单关键词模式 / 关键词组模式
- 自动检测并配置 Python 3.12 + CUDA 环境
- Web UI 界面，浏览器操作

## 快速开始

### 1. 安装环境

双击 `setup.bat`，自动完成：
- 检测 NVIDIA 驱动
- 安装 Python 3.12（如需要）
- 创建虚拟环境
- 安装 CUDA 版 PyTorch + 所有依赖

### 2. 启动

双击 `start.bat`，浏览器自动打开 `http://localhost:7860`

## 文件说明

| 文件 | 说明 |
|------|------|
| `setup.bat` | 双击安装环境 |
| `setup.py` | 自动安装脚本 |
| `start.bat` | 双击启动程序 |
| `start.py` | 启动入口 |
| `ai_classifier_ui.py` | Web UI 主程序 |
| `ai_image_classifier.py` | 命令行版本 |

## 系统要求

- Windows 10/11
- NVIDIA 显卡 + 驱动（运行 `nvidia-smi` 确认）
- 约 10GB 磁盘空间（Python + PyTorch + 模型权重）

## 阈值说明

| 模型 | 类型 | 建议阈值 |
|------|------|---------|
| CLIP | 余弦相似度 (-1~1) | 0.10 ~ 0.25 |
| SigLIP | 概率分布 (总和=1) | 0.01 ~ 0.10 |

SigLIP 会自动计算 chance level × 1.2 作为有效阈值下限。

## 支持格式

jpg, jpeg, png, bmp, webp, gif, tiff, tif, jfif, heic, heif

不支持的格式（视频、RAW 等）会自动复制到 `output/unsupported/` 目录。
