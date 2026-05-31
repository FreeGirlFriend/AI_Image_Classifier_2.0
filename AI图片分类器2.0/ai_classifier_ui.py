#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 图片关键词分类器 — Gradio Web UI
启动方式: python ai_classifier_ui.py
"""

import os
import sys
import json
import shutil
import atexit
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import Counter

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import gradio as gr

# ── 常量 ─────────────────────────────────────────────────────────
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff", ".tif", ".jfif"}
PARAMS_FILE = Path(__file__).parent / ".last_params.json"

MODELS = {
    "CLIP ViT-L-14-336": {
        "id": "ViT-L-14-336",
        "pretrained": "openai",
        "type": "clip",
        "suggested_threshold": 0.15,
        "desc": "余弦相似度，范围 -1~1，建议 0.10~0.25 | 分辨率 336px",
    },
    "SigLIP ViT-L-14-384": {
        "id": "ViT-L-14-SigLIP-384",
        "hf_id": "google/siglip-base-patch16-384",
        "type": "siglip",
        "suggested_threshold": 0.01,
        "desc": "概率分布(总和=1)，建议 0.01~0.10",
    },
}

# ── 并行图片加载 (ThreadPool，避免 Windows spawn 问题) ───────────
from concurrent.futures import ThreadPoolExecutor as _ThreadPool


def _load_and_preprocess(path, preprocess):
    """单张图片加载+预处理，供线程池调用"""
    try:
        from PIL import Image
        return preprocess(Image.open(path).convert("RGB"))
    except Exception:
        import torch
        return torch.zeros(3, 336, 336)


# ── 全局退出标志 ─────────────────────────────────────────────────
_server_ref = None


def _cleanup():
    """进程退出时清理"""
    try:
        if _server_ref:
            _server_ref.close()
    except Exception:
        pass


atexit.register(_cleanup)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                          工具函数                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def require_cuda():
    """强制要求 CUDA，不可用则抛异常"""
    if not cuda_ok():
        raise RuntimeError(
            "CUDA 不可用！请安装 CUDA 版 PyTorch:\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "如果已安装仍报错，请检查 NVIDIA 驱动和 CUDA Toolkit。"
        )


def gpu_info():
    if not cuda_ok():
        return "❌ CUDA 不可用"
    import torch
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"GPU: {name} ({mem:.1f} GB)"


def safe_dirname(s: str) -> str:
    for c in '<>:"/\\|?*':
        s = s.replace(c, "_")
    return s.strip()


def scan_images(src_dir: str) -> Tuple[List[str], List[str]]:
    """扫描目录，返回 (支持的图片, 不支持的文件)"""
    root = Path(src_dir)
    if not root.exists():
        return [], []
    supported = set()
    unsupported = []
    for f in root.rglob("*"):
        if not f.is_file() or not f.suffix:
            continue
        if f.suffix.lower() in SUPPORTED_EXT:
            supported.add(str(f))
        else:
            unsupported.append(str(f))
    return sorted(supported), sorted(unsupported)


def save_params(params: dict):
    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def load_params() -> Optional[dict]:
    if not PARAMS_FILE.exists():
        return None
    try:
        with open(PARAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        分类器实现 (CUDA)                         ║
# ╚══════════════════════════════════════════════════════════════════╝

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_siglip_pipe = None


CLIP_MODEL_NAME = "ViT-L-14-336"
CLIP_PRETRAINED = "openai"


def _get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_tokenizer
    require_cuda()
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to("cuda").eval().half()  # fp16 强制
    _clip_model, _clip_preprocess, _clip_tokenizer = model, preprocess, tokenizer
    torch.cuda.empty_cache()
    return model, preprocess, tokenizer


def _get_siglip():
    global _siglip_pipe
    if _siglip_pipe is not None:
        return _siglip_pipe
    require_cuda()
    from transformers import pipeline as hf_pipe
    _siglip_pipe = hf_pipe(
        "zero-shot-image-classification",
        model="google/siglip-base-patch16-384",
        device=0,  # 强制 GPU 0
    )
    return _siglip_pipe


def classify_clip(images: List[str], keywords: List[str], threshold: float,
                  progress=gr.Progress()) -> Tuple[List[dict], dict]:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    model, preprocess, tokenizer = _get_clip()

    # 增强提示词
    templates = [
        "a photo of {}", "a close-up photo of {}",
        "a detailed photo showing {}", "an image containing {}",
    ]
    all_texts = []
    for kw in keywords:
        for t in templates:
            all_texts.append(t.format(kw))
        all_texts.append(kw)
    all_texts = list(set(all_texts))

    text_tokens = tokenizer(all_texts).to("cuda")
    with torch.no_grad():
        text_feat = F.normalize(model.encode_text(text_tokens), dim=-1)

    kw_to_idx = {
        kw: [i for i, t in enumerate(all_texts) if kw in t or t == kw]
        for kw in keywords
    }

    # 并行加载 + batch 推理
    batch_size = 256
    results = []
    stats = {"total": len(images), "hit": 0, "miss": 0, "error": 0}
    done = 0

    with _ThreadPool(max_workers=8) as pool:
        for batch_start in range(0, len(images), batch_size):
            batch_paths = images[batch_start:batch_start + batch_size]

            # 并行加载图片
            tensors_list = list(pool.map(
                lambda p: _load_and_preprocess(p, preprocess), batch_paths
            ))

            tensors = torch.stack(tensors_list).to("cuda", non_blocking=True).half()

            with torch.no_grad():
                img_feat = F.normalize(model.encode_image(tensors), dim=-1)
                sims = (img_feat @ text_feat.T).float().cpu().numpy()
                sims = np.clip(sims, 0, 1)

            for i, img_path in enumerate(batch_paths):
                sim = sims[i]
                kw_scores = {
                    kw: float(np.max(sim[kw_to_idx[kw]])) if kw_to_idx[kw] else 0.0
                    for kw in keywords
                }
                matched = [kw for kw in keywords if kw_scores[kw] >= threshold]
                best_kw = max(kw_scores, key=kw_scores.get)
                results.append({
                    "path": img_path,
                    "filename": os.path.basename(img_path),
                    "matched": matched,
                    "scores": kw_scores,
                    "best_kw": best_kw,
                    "best_score": kw_scores[best_kw],
                })
                stats["hit" if matched else "miss"] += 1

            done += len(batch_paths)
            progress(done / len(images), desc=f"CLIP 识别中 {done}/{len(images)}")

    stats["error"] = len(images) - stats["hit"] - stats["miss"]
    return results, stats


def classify_siglip(images: List[str], keywords: List[str], threshold: float,
                    progress=gr.Progress()) -> Tuple[List[dict], dict]:
    from PIL import Image

    pipe = _get_siglip()
    results = []
    stats = {"total": len(images), "hit": 0, "miss": 0, "error": 0}

    chance = 1.0 / len(keywords) if keywords else 0.01
    eff_threshold = max(threshold, chance * 1.2)

    for i, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert("RGB")
            preds = pipe(img, candidate_labels=keywords)
            scores = {p["label"]: p["score"] for p in preds}
        except Exception:
            scores = {kw: 0.0 for kw in keywords}
            stats["error"] += 1

        matched = [kw for kw in keywords if scores.get(kw, 0) >= eff_threshold]
        best_kw = max(scores, key=scores.get) if scores else (keywords[0] if keywords else "")
        results.append({
            "path": img_path,
            "filename": os.path.basename(img_path),
            "matched": matched,
            "scores": scores,
            "best_kw": best_kw,
            "best_score": scores.get(best_kw, 0),
        })
        stats["hit" if matched else "miss"] += 1
        progress((i + 1) / len(images), desc=f"SigLIP 识别中 {i+1}/{len(images)}")

    return results, stats


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     Gradio 回调函数                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def on_model_change(model_name):
    info = MODELS.get(model_name, MODELS["CLIP ViT-L-14-336"])
    return gr.update(value=info["suggested_threshold"], label=f"阈值 ({info['desc']})")


def do_classify(src_dir, out_dir, mode, keywords_str, groups_str,
                model_name, threshold, custom_pkl=None, custom_prefilter=None,
                progress=gr.Progress()):
    if not src_dir or not Path(src_dir).exists():
        return "❌ 源目录不存在", None, ""

    keywords = []
    groups = None

    if mode == "关键词组模式":
        if not groups_str or not groups_str.strip():
            return "❌ 请输入关键词组", None, ""
        groups = {}
        for line in groups_str.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            folder, kws = line.split(":", 1)
            kws_list = [k.strip() for k in kws.split(",") if k.strip()]
            if kws_list:
                groups[folder.strip()] = kws_list
                keywords.extend(kws_list)
        if not groups:
            return "❌ 关键词组格式错误 (格式: 文件夹名:关键词1,关键词2)", None, ""
    elif mode == "自定义分类器模式":
        # 自定义分类器模式：关键词可选（用于预筛选）
        if custom_prefilter and custom_prefilter.strip():
            keywords = [k.strip() for k in custom_prefilter.split(",") if k.strip()]
    else:
        if not keywords_str or not keywords_str.strip():
            return "❌ 请输入关键词", None, ""
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

    if not keywords and mode != "自定义分类器模式":
        return "❌ 关键词为空", None, ""

    if not out_dir or not out_dir.strip():
        out_dir = str(Path(src_dir).parent / "classified_output")

    images, unsupported = scan_images(src_dir)
    if not images and not unsupported:
        return "❌ 未找到任何文件", None, ""
    if not images:
        return "❌ 未找到支持格式的图片（仅有视频/RAW等不支持格式）", None, ""

    # CUDA 检查
    if not cuda_ok():
        return "❌ CUDA 不可用，请先安装 CUDA 版 PyTorch", None, ""

    params = {
        "source": src_dir,
        "output": out_dir,
        "keywords": keywords,
        "groups": groups,
        "threshold": threshold,
        "model_name": model_name,
    }
    save_params(params)

    model_info = MODELS[model_name]
    results = []
    stats = {"total": len(images), "hit": 0, "miss": 0, "error": 0}

    # 关键词模式：先用 CLIP/SigLIP 分类
    if mode != "自定义分类器模式":
        if model_info["type"] == "clip":
            results, stats = classify_clip(images, keywords, threshold, progress)
        else:
            results, stats = classify_siglip(images, keywords, threshold, progress)

    all_kw = list(groups.keys()) if groups else keywords
    cls_dir = Path(out_dir) / "classified"
    unc_dir = Path(out_dir) / "unclassified"
    cls_dir.mkdir(parents=True, exist_ok=True)
    unc_dir.mkdir(parents=True, exist_ok=True)

    # ── 关键词 → 文件夹映射 ──
    kw_to_folder = {}
    if groups:
        for folder, kws in groups.items():
            for kw in kws:
                kw_to_folder[kw] = folder

    # ── 自定义分类器模式 ──
    custom_model_data = None
    is_custom_mode = (mode == "自定义分类器模式")

    if is_custom_mode:
        if not custom_pkl or not os.path.isfile(custom_pkl):
            return "❌ 请选择有效的 .pkl 模型文件", None, ""
        try:
            import pickle as _pkl
            with open(custom_pkl, "rb") as f:
                custom_model_data = _pkl.load(f)
        except Exception as e:
            return f"❌ 模型加载失败: {e}", None, ""

        # 有预筛选关键词：先用 CLIP 筛选
        if keywords:
            model_info = MODELS[model_name]
            progress(0.1, desc="CLIP 预筛选中...")
            if model_info["type"] == "clip":
                results, stats = classify_clip(images, keywords, threshold, progress)
            else:
                results, stats = classify_siglip(images, keywords, threshold, progress)
            # 只保留匹配的图片
            images_for_custom = [r["path"] for r in results if r["matched"]]
            if not images_for_custom:
                return "❌ 预筛选后无匹配图片", None, ""
            progress(0.7, desc=f"预筛选完成，{len(images_for_custom)} 张匹配")
        else:
            # 无预筛选：对所有图片分类
            images_for_custom = images
            stats = {"total": len(images), "hit": len(images), "miss": 0, "error": 0}
            results = [{"path": p, "filename": os.path.basename(p), "matched": [], "scores": {}, "best_kw": "-", "best_score": 0} for p in images]

        # 运行自定义分类器
        from custom_classifier_trainer import predict_batch
        def _custom_progress(frac, desc=""):
            progress(0.7 + frac * 0.2, desc=desc or f"自定义分类 {int(frac*100)}%")

        custom_results = predict_batch(images_for_custom, custom_pkl, progress=_custom_progress)
        custom_map = {cr["path"]: cr["category"] for cr in custom_results if "error" not in cr}
        for r in results:
            if r["path"] in custom_map:
                r["custom_cat"] = custom_map[r["path"]]

    # ── 建目录 ──
    kw_dirs = {}
    custom_cat_dirs = {}

    if is_custom_mode and custom_model_data:
        # 自定义分类器模式：按类别建目录
        cats = custom_model_data["categories"]
        if keywords:
            # 有预筛选：keyword/category 两层
            for kw in keywords:
                kw_d = cls_dir / safe_dirname(kw)
                kw_d.mkdir(parents=True, exist_ok=True)
                kw_dirs[kw] = kw_d
                for cat in cats:
                    sub = kw_d / safe_dirname(cat)
                    sub.mkdir(parents=True, exist_ok=True)
                    custom_cat_dirs[(kw, cat)] = sub
        else:
            # 无预筛选：直接按类别
            for cat in cats:
                d = cls_dir / safe_dirname(cat)
                d.mkdir(parents=True, exist_ok=True)
                custom_cat_dirs[("_all", cat)] = d
    elif groups:
        for folder, kws in groups.items():
            d = cls_dir / safe_dirname(folder)
            d.mkdir(parents=True, exist_ok=True)
            for kw in kws:
                kw_dirs[kw] = d
    else:
        for kw in keywords:
            d = cls_dir / safe_dirname(kw)
            d.mkdir(parents=True, exist_ok=True)
            kw_dirs[kw] = d

    # ── 复制文件 ──
    file_ops = []
    for r in results:
        matched = r["matched"]
        has_custom = "custom_cat" in r

        if is_custom_mode and has_custom:
            cat = r["custom_cat"]
            if keywords and matched:
                # 有预筛选：放到 keyword/category
                best_m = max(matched, key=lambda kw: r["scores"].get(kw, 0))
                d = custom_cat_dirs.get((best_m, cat))
                if d:
                    file_ops.append((r["path"], d))
                    continue
            elif not keywords:
                # 无预筛选：直接放到 category
                d = custom_cat_dirs.get(("_all", cat))
                if d:
                    file_ops.append((r["path"], d))
                    continue
            # fallback
            file_ops.append((r["path"], unc_dir))
        elif matched:
            if groups:
                seen_dirs = set()
                for kw in matched:
                    d = kw_dirs.get(kw)
                    if d and str(d) not in seen_dirs:
                        seen_dirs.add(str(d))
                        file_ops.append((r["path"], d))
            else:
                best_m = max(matched, key=lambda kw: r["scores"].get(kw, 0))
                d = kw_dirs.get(best_m)
                if d:
                    file_ops.append((r["path"], d))
        else:
            file_ops.append((r["path"], unc_dir))

    total_copy = len(file_ops) + len(unsupported)
    copied = 0
    for src_path, tgt_dir in file_ops:
        basename = os.path.basename(src_path)
        dest = tgt_dir / basename
        if dest.exists():
            name, ext = os.path.splitext(basename)
            dest = tgt_dir / f"{name}_{datetime.now().strftime('%H%M%S%f')}{ext}"
        shutil.copy2(src_path, dest)
        copied += 1
        if copied % 50 == 0 or copied == len(file_ops):
            progress(copied / total_copy, desc=f"复制文件 {copied}/{total_copy}")

    # 复制不支持格式的文件到 unsupported 文件夹
    unsup_copied = 0
    if unsupported:
        unsup_dir = Path(out_dir) / "unsupported"
        unsup_dir.mkdir(parents=True, exist_ok=True)
        for src_path in unsupported:
            basename = os.path.basename(src_path)
            dest = unsup_dir / basename
            if dest.exists():
                name, ext = os.path.splitext(basename)
                dest = unsup_dir / f"{name}_{datetime.now().strftime('%H%M%S%f')}{ext}"
            shutil.copy2(src_path, dest)
            unsup_copied += 1
            copied += 1
            if copied % 50 == 0:
                progress(copied / total_copy, desc=f"复制文件 {copied}/{total_copy}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "threshold": threshold,
        "keywords": keywords,
        "groups": groups,
        "stats": stats,
        "mode": mode,
        "custom_classifier": custom_pkl if is_custom_mode else None,
    }
    report_path = Path(out_dir) / "classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    kw_cnt = Counter()
    for r in results:
        for kw in r["matched"]:
            kw_cnt[kw] += 1

    import torch
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_used = torch.cuda.memory_allocated(0) / 1024**3
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    lines = [
        f"✅ 分类完成",
        f"",
        f"设备: {gpu_name} (CUDA fp16)",
        f"显存: {gpu_mem_used:.1f} / {gpu_mem_total:.1f} GB",
    ]
    if is_custom_mode:
        lines.append(f"模式: 自定义分类器")
        lines.append(f"模型: {os.path.basename(custom_pkl)}")
        if keywords:
            lines.append(f"预筛选: {', '.join(keywords)}")
    else:
        lines.append(f"模型: {model_name}")
    lines.extend([
        f"扫描: {stats['total'] + unsup_copied} 个文件",
        f"已分析: {stats['total']} 张图片",
        f"  命中: {stats['hit']} 张",
        f"  未分类: {stats['miss']} 张",
        f"  错误: {stats['error']} 张",
        f"  命中率: {stats['hit']/stats['total']*100:.1f}%",
    ])
    if unsup_copied > 0:
        lines.append(f"跳过: {unsup_copied} 个不支持格式 (已复制到 unsupported/)")
    lines.extend([
        f"已复制: {copied} 个文件",
        f"报告: {report_path}",
    ])
    if kw_cnt:
        lines.append("")
        lines.append("关键词命中:")
        for kw, cnt in kw_cnt.most_common():
            lines.append(f"  {kw}: {cnt} 张")

    # 自定义分类器子类别统计
    if custom_model_data:
        custom_cat_cnt = Counter()
        for r in results:
            if "custom_cat" in r:
                custom_cat_cnt[r["custom_cat"]] += 1
        if custom_cat_cnt:
            lines.append("")
            lines.append("自定义分类器子类别:")
            for cat, cnt in custom_cat_cnt.most_common():
                lines.append(f"  {cat}: {cnt} 张")

    # 限制表格行数，避免内存爆炸
    MAX_TABLE_ROWS = 500
    table_data = []
    for r in results[:MAX_TABLE_ROWS]:
        matched_str = ", ".join(r["matched"]) if r["matched"] else "-"
        row = [r["filename"], matched_str, f"{r['best_score']:.3f}", r["best_kw"]]
        if custom_model_data:
            row.append(r.get("custom_cat", "-"))
        table_data.append(row)
    if len(results) > MAX_TABLE_ROWS:
        extra = ["...", f"(共 {len(results)} 条，仅显示前 {MAX_TABLE_ROWS} 条)", "", ""]
        if custom_model_data:
            extra.append("")
        table_data.append(extra)

    summary = "\n".join(lines)
    return summary, table_data, out_dir


def do_diagnose(model_name, progress=gr.Progress()):
    """模型功能性自检 — 只检测模型本身，不检测环境"""
    lines = []
    model_info = MODELS[model_name]

    lines.append(f"═══ {model_name} 功能自检 ═══")

    # 前置检查
    if not cuda_ok():
        lines.append("  ❌ CUDA 不可用，请先到「环境依赖」Tab 修复")
        return "\n".join(lines)

    import torch
    lines.append(f"  设备: {torch.cuda.get_device_name(0)} (CUDA fp16)")
    lines.append("")

    try:
        if model_info["type"] == "clip":
            model, preprocess, tokenizer = _get_clip()
            import torch.nn.functional as F
            import numpy as np
            from PIL import Image

            param_count = sum(p.numel() for p in model.parameters())
            input_res = preprocess.transforms[0].size if hasattr(preprocess.transforms[0], 'size') else "336"
            lines.append(f"  模型: {CLIP_MODEL_NAME} | 参数量: {param_count/1e6:.0f}M | 精度: fp16 | batch: 256")
            lines.append(f"  输入分辨率: {input_res} | 预训练: {CLIP_PRETRAINED}")
            lines.append("")

            # 推理测试
            print("  [自检] 推理测试...")
            test_img = Image.new("RGB", (336, 336), (139, 69, 19))
            labels = ["brown color", "blue color", "cat"]
            tokens = tokenizer(labels).to("cuda")
            with torch.no_grad():
                txt = F.normalize(model.encode_text(tokens), dim=-1)
                img_t = preprocess(test_img).unsqueeze(0).to("cuda").half()
                img_f = F.normalize(model.encode_image(img_t), dim=-1)
                s = (img_f @ txt.T).float().cpu().numpy().flatten()
            scores = dict(zip(labels, [float(x) for x in s]))
            lines.append(f"  推理: {', '.join(f'{k}={v:.4f}' for k, v in scores.items())}")

            if max(scores.values()) < 0.001:
                lines.append("  ❌ 分数异常低，模型可能损坏")
                return "\n".join(lines)
            lines.append("  ✓ 推理正常")

            # 区分测试
            print("  [自检] 区分测试...")
            brown = Image.new("RGB", (336, 336), (139, 69, 19))
            blue = Image.new("RGB", (336, 336), (0, 0, 200))
            labels2 = ["brown color", "blue color"]
            tokens2 = tokenizer(labels2).to("cuda")
            with torch.no_grad():
                txt2 = F.normalize(model.encode_text(tokens2), dim=-1)
                b_t = preprocess(brown).unsqueeze(0).to("cuda").half()
                u_t = preprocess(blue).unsqueeze(0).to("cuda").half()
                b_f = F.normalize(model.encode_image(b_t), dim=-1)
                u_f = F.normalize(model.encode_image(u_t), dim=-1)
                b_s = (b_f @ txt2.T).float().cpu().numpy().flatten()
                u_s = (u_f @ txt2.T).float().cpu().numpy().flatten()
            lines.append(f"  棕色→brown:{b_s[0]:.4f} blue:{b_s[1]:.4f}")
            lines.append(f"  蓝色→brown:{u_s[0]:.4f} blue:{u_s[1]:.4f}")
            if b_s[0] > b_s[1] and u_s[1] > u_s[0]:
                lines.append("  ✓ 区分能力正常")
            else:
                lines.append("  ⚠ 区分能力较弱")

            # 显存占用
            lines.append(f"\n  显存占用: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

        else:  # SigLIP
            pipe = _get_siglip()
            from PIL import Image

            # 推理测试
            print("  [自检] 推理测试...")
            test_img = Image.new("RGB", (384, 384), (139, 69, 19))
            labels = ["brown color", "blue color", "cat"]
            preds = pipe(test_img, candidate_labels=labels)
            scores = {p["label"]: p["score"] for p in preds}
            lines.append(f"  推理: {', '.join(f'{k}={v:.4f}' for k, v in scores.items())}")

            if max(scores.values()) < 0.001:
                lines.append("  ❌ 所有分数接近 0，模型异常")
                return "\n".join(lines)
            elif len(set(round(v, 4) for v in scores.values())) == 1:
                lines.append("  ❌ 所有分数相同，无区分能力")
                return "\n".join(lines)
            lines.append("  ✓ 推理正常")

            # 区分测试
            print("  [自检] 区分测试...")
            brown = Image.new("RGB", (384, 384), (139, 69, 19))
            blue = Image.new("RGB", (384, 384), (0, 0, 200))
            labels2 = ["brown color", "blue color"]
            b_preds = pipe(brown, candidate_labels=labels2)
            u_preds = pipe(blue, candidate_labels=labels2)
            b_scores = {p["label"]: p["score"] for p in b_preds}
            u_scores = {p["label"]: p["score"] for p in u_preds}
            lines.append(f"  棕色→{', '.join(f'{k}={v:.4f}' for k, v in b_scores.items())}")
            lines.append(f"  蓝色→{', '.join(f'{k}={v:.4f}' for k, v in u_scores.items())}")
            b_best = max(b_scores, key=b_scores.get)
            u_best = max(u_scores, key=u_scores.get)
            if b_best == "brown color" and u_best == "blue color":
                lines.append("  ✓ 区分能力正常")
            else:
                lines.append(f"  ⚠ 区分能力较弱 (棕色→{b_best}, 蓝色→{u_best})")

            # 显存占用
            lines.append(f"\n  显存占用: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

        # 网络检查
        lines.append(f"\n═══ 网络检查 ═══")
        import urllib.request
        for name, url in [("HuggingFace", "https://huggingface.co"), ("HF Mirror", "https://hf-mirror.com")]:
            try:
                urllib.request.urlopen(url, timeout=8)
                lines.append(f"  ✓ {name} 连通")
            except Exception:
                lines.append(f"  ✗ {name} 不通")

        lines.append(f"\n✅ {model_name} 自检通过")

    except Exception as e:
        lines.append(f"\n❌ 自检失败: {e}")
        import traceback
        lines.append(traceback.format_exc())

    return "\n".join(lines)


def do_check_env(progress=gr.Progress()):
    """环境依赖检查与自动修复"""
    import subprocess

    lines = []
    fixed = 0  # 记录修复了几个问题

    # ═══════════════════════════════════════════════════
    # 第一步: NVIDIA 驱动检测
    # ═══════════════════════════════════════════════════
    lines.append("═══ [1/5] NVIDIA 驱动检测 ═══")
    driver_ok = False
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            for line in r.stdout.split("\n"):
                if "Driver Version" in line:
                    lines.append(f"  {line.strip()}")
                    break
            lines.append("  ✓ nvidia-smi 正常")
            driver_ok = True
        else:
            lines.append("  ❌ nvidia-smi 执行失败")
            lines.append("  → 请从 https://www.nvidia.com/drivers 下载安装最新驱动")
    except FileNotFoundError:
        lines.append("  ❌ nvidia-smi 未找到")
        lines.append("  → 请从 https://www.nvidia.com/drivers 下载安装最新驱动")
        lines.append("  → 安装完成后重启电脑，再回来重新检测")
    lines.append("")

    # ═══════════════════════════════════════════════════
    # 第二步: Python 依赖检查 + 自动安装
    # ═══════════════════════════════════════════════════
    lines.append("═══ [2/5] Python 依赖检查 ═══")
    deps = [
        ("torch",       "torch+cuda",  ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"]),
        ("open_clip",   "open-clip-torch", None),
        ("transformers","transformers", None),
        ("PIL",         "Pillow",       None),
        ("numpy",       "numpy",        None),
        ("sklearn",     "scikit-learn", None),
        ("tqdm",        "tqdm",         None),
        ("gradio",      "gradio",       None),
    ]

    missing = []
    for mod, pip_name, args in deps:
        try:
            __import__(mod)
            lines.append(f"  ✓ {mod}")
        except ImportError:
            lines.append(f"  ✗ {mod} (缺失)")
            missing.append((pip_name, args))

    if missing:
        lines.append(f"\n  缺失 {len(missing)} 个依赖，自动安装中...\n")
        for i, (pip_name, args) in enumerate(missing):
            progress((i + 1) / (len(missing) + 4), desc=f"安装 {pip_name}")
            cmd = [sys.executable, "-m", "pip", "install"]
            if args:
                cmd.extend(args)
            else:
                cmd.append(pip_name)
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode == 0:
                lines.append(f"  ✓ {pip_name} 安装成功")
                fixed += 1
            else:
                lines.append(f"  ✗ {pip_name} 安装失败")
                lines.append(f"    {(r.stderr or r.stdout)[-200:]}")
    else:
        lines.append("  所有依赖已就绪 ✓")
    lines.append("")

    # ═══════════════════════════════════════════════════
    # 第三步: CUDA + PyTorch 检测与自动修复
    # ═══════════════════════════════════════════════════
    lines.append("═══ [3/5] CUDA + PyTorch 检测与修复 ═══")
    cuda_ready = False

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    lines.append(f"  Python: {py_ver} ({sys.executable})")

    # Python 3.14+ 没有 CUDA 版 PyTorch
    if sys.version_info >= (3, 14):
        lines.append(f"\n  ⚠ Python {py_ver} 暂无 CUDA 版 PyTorch!")
        lines.append(f"  → 需要使用 Python 3.12 运行本程序")
        lines.append(f"  → 请用以下命令启动:")
        lines.append(f"    D:\\PY\\JS\\venv\\Scripts\\python.exe D:\\PY\\JS\\ai_classifier_ui.py")

    try:
        import torch
        lines.append(f"  PyTorch: {torch.__version__}")
        lines.append(f"  CUDA 编译版本: {torch.version.cuda}")

        # 情况1: PyTorch 是 CPU 版 (torch.version.cuda is None)
        if torch.version.cuda is None:
            lines.append(f"\n  ❌ 当前 PyTorch 是 CPU 版本，无法使用 GPU!")
            if sys.version_info >= (3, 14):
                lines.append(f"  → Python {py_ver} 不支持 CUDA 版 PyTorch")
                lines.append(f"  → 请使用 venv 启动: D:\\PY\\JS\\venv\\Scripts\\python.exe D:\\PY\\JS\\ai_classifier_ui.py")
            else:
                lines.append(f"  → 自动卸载 CPU 版，安装 CUDA 版...")
                progress(0.5, desc="重装 PyTorch CUDA 版")
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "-y"],
                              capture_output=True)
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "torch", "torchvision",
                     "--index-url", "https://download.pytorch.org/whl/cu124"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace"
                )
                if r.returncode == 0:
                    lines.append(f"  ✓ PyTorch CUDA 版安装成功，需重启程序")
                    fixed += 1
                else:
                    lines.append(f"  ✗ 安装失败: {(r.stderr or r.stdout)[-200:]}")

        # 情况2: 有 CUDA 编译但运行时不可用
        elif not torch.cuda.is_available():
            lines.append(f"\n  ⚠ PyTorch 有 CUDA {torch.version.cuda}，但运行时不可用")
            # 检查驱动
            if not driver_ok:
                lines.append(f"  → 原因: NVIDIA 驱动未安装或版本太旧")
                lines.append(f"  → 请安装/更新驱动后重启电脑")
            else:
                lines.append(f"  → 可能原因: 驱动 CUDA 版本低于 PyTorch 编译版本")
                lines.append(f"  → 尝试重装匹配当前驱动的 PyTorch...")
                progress(0.5, desc="重装 PyTorch")
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "-y"],
                              capture_output=True)
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "torch", "torchvision",
                     "--index-url", "https://download.pytorch.org/whl/cu124"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace"
                )
                if r.returncode == 0:
                    lines.append(f"  ✓ 重装完成，请重启程序")
                    fixed += 1
                else:
                    lines.append(f"  ✗ 重装失败")
                    lines.append(f"  → 请手动执行: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

        # 情况3: CUDA 正常
        else:
            lines.append(f"  CUDA 运行时: ✓ 可用")
            lines.append(f"  cuDNN: {torch.backends.cudnn.version()} (启用: {torch.backends.cudnn.enabled})")
            if not torch.backends.cudnn.enabled:
                lines.append(f"  ⚠ cuDNN 未启用，自动开启...")
                torch.backends.cudnn.enabled = True
                fixed += 1

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                lines.append(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

            # 显存压力测试
            lines.append(f"  显存测试中...")
            try:
                test_t = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
                _ = test_t @ test_t.T
                torch.cuda.synchronize()
                del test_t
                torch.cuda.empty_cache()
                lines.append(f"  ✓ GPU 计算正常 (fp16)")
                cuda_ready = True
            except Exception as e:
                lines.append(f"  ❌ GPU 计算异常: {e}")

    except ImportError:
        lines.append(f"  ❌ PyTorch 未安装 (上一步应已安装)")
        lines.append(f"  → 请手动执行: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    lines.append("")

    # ═══════════════════════════════════════════════════
    # 第四步: 模型权重可访问性检测
    # ═══════════════════════════════════════════════════
    lines.append("═══ [4/5] 网络连通性检测 ═══")
    import urllib.request

    hf_ok = False
    mirror_ok = False
    for name, url in [("HuggingFace", "https://huggingface.co"), ("HF Mirror", "https://hf-mirror.com")]:
        try:
            urllib.request.urlopen(url, timeout=8)
            lines.append(f"  ✓ {name} 连通")
            if "huggingface" in name.lower():
                hf_ok = True
            else:
                mirror_ok = True
        except Exception:
            lines.append(f"  ✗ {name} 不通")

    if not hf_ok and not mirror_ok:
        lines.append(f"  ⚠ 所有镜像不通，模型无法下载")
        lines.append(f"  → 请检查网络或设置代理:")
        lines.append(f"    set HF_ENDPOINT=https://hf-mirror.com")
    elif not hf_ok and mirror_ok:
        lines.append(f"  → HuggingFace 不通，自动使用 HF Mirror")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        fixed += 1

    # ═══════════════════════════════════════════════════
    # 第五步: 模型权重缓存检测 + 自动下载
    # ═══════════════════════════════════════════════════
    lines.append(f"\n═══ [5/5] 模型权重缓存检测 ═══")
    models_to_check = [
        (f"CLIP {CLIP_MODEL_NAME}", "open_clip", CLIP_MODEL_NAME, CLIP_PRETRAINED),
        ("SigLIP ViT-L-14-384", "transformers", "google/siglip-base-patch16-384", None),
    ]
    models_missing = []

    for display_name, lib, model_id, pretrained in models_to_check:
        cached = False
        try:
            if lib == "open_clip":
                import open_clip
                # 尝试获取预训练路径
                from open_clip.pretrained import get_pretrained_url
                url = get_pretrained_url(model_id, pretrained)
                cached = url is not None
            else:
                from huggingface_hub import try_to_load_from_cache
                path = try_to_load_from_cache(model_id, "model.safetensors")
                cached = path is not None and path != "None"
        except Exception:
            pass

        # 二次检测：直接查缓存目录
        if not cached:
            import glob
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
            if lib == "open_clip":
                pattern = os.path.join(cache_dir, "**", "*CLIP-ViT-L-14-336*")
            else:
                pattern = os.path.join(cache_dir, "**", "*siglip*")
            cached = bool(glob.glob(pattern, recursive=True))

        if cached:
            lines.append(f"  ✓ {display_name} 权重已缓存")
        else:
            lines.append(f"  ✗ {display_name} 权重未缓存 (首次使用需下载)")
            models_missing.append((display_name, lib, model_id, pretrained))

    if models_missing:
        lines.append(f"\n  需要下载 {len(models_missing)} 个模型权重，开始自动下载...")
        for i, (display_name, lib, model_id, pretrained) in enumerate(models_missing):
            progress(0.7 + 0.25 * (i / len(models_missing)), desc=f"下载 {display_name}")
            lines.append(f"  → 正在下载 {display_name}...")
            try:
                if lib == "open_clip":
                    import open_clip, gc
                    # 触发下载（会缓存到本地）
                    m, _, _ = open_clip.create_model_and_transforms(model_id, pretrained=pretrained, device="cpu")
                    del m; gc.collect()
                else:
                    from transformers import pipeline as hf_pipe
                    import gc
                    p = hf_pipe("zero-shot-image-classification", model=model_id, device="cpu")
                    del p; gc.collect()
                lines.append(f"  ✓ {display_name} 下载完成")
                fixed += 1
            except Exception as e:
                lines.append(f"  ✗ {display_name} 下载失败: {e}")
                if not hf_ok and not mirror_ok:
                    lines.append(f"    → 原因: 网络不通，请设置代理后重试")
                elif not hf_ok:
                    lines.append(f"    → 请设置环境变量: set HF_ENDPOINT=https://hf-mirror.com")
    else:
        lines.append(f"  所有模型权重已缓存 ✓")

    # ── 汇总 ──
    lines.append(f"\n{'═'*40}")
    if cuda_ready and not missing and not models_missing:
        lines.append(f"✅ 环境就绪，全部正常！可直接使用 GPU 分析")
    elif fixed > 0:
        lines.append(f"🔧 修复了 {fixed} 个问题")
        if not cuda_ready:
            lines.append(f"⚠ CUDA 仍未就绪，可能需要重启程序")
        elif models_missing:
            lines.append(f"⚠ 部分模型权重下载失败，请检查网络后重试")
        else:
            lines.append(f"✅ CUDA 已就绪")
    else:
        lines.append(f"⚠ 存在未解决的问题，请按提示操作")

    return "\n".join(lines)


def do_load_last_config():
    params = load_params()
    if not params:
        return "", "", "普通单关键词模式", "", "CLIP ViT-L-14-336", 0.15, "没有找到上次的配置"

    model_name = params.get("model_name", "CLIP ViT-L-14-336")
    if model_name not in MODELS:
        model_name = "CLIP ViT-L-14-336"

    summary = f"上次配置: {params.get('source', '?')} → {params.get('output', '?')}"
    return (
        params.get("source", ""),
        params.get("output", ""),
        "关键词组模式" if params.get("groups") else "普通单关键词模式",
        ", ".join(params.get("keywords", [])),
        model_name,
        params.get("threshold", MODELS[model_name]["suggested_threshold"]),
        summary,
    )


def do_exit():
    """优雅退出：先返回提示，再延迟杀进程"""
    import torch
    global _clip_model, _clip_preprocess, _clip_tokenizer, _siglip_pipe

    # 清理 GPU 显存
    _clip_model = None
    _clip_preprocess = None
    _clip_tokenizer = None
    _siglip_pipe = None
    if cuda_ok():
        torch.cuda.empty_cache()

    # 延迟 1 秒后杀进程，让前端收到返回值
    def _kill():
        import time
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=_kill, daemon=True).start()

    return "正在关闭..."


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        Gradio UI                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def build_ui():
    with gr.Blocks(title="AI 图片分类器") as app:
        gr.Markdown("# 🖼️ AI 图片关键词分类器 v2.0\nCLIP + SigLIP · 零样本识别 · CUDA fp16", elem_classes="main-title")

        # GPU 状态栏
        gpu_status = gr.Markdown(f"**设备:** {gpu_info()}", elem_classes="info-box")

        with gr.Tabs():
            # ── Tab 1: 分类 ──
            with gr.TabItem("🔍 开始分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        src_dir = gr.Textbox(label="📁 源图片目录", placeholder="D:\\photos")
                        out_dir = gr.Textbox(label="📂 输出目录", placeholder="留空则自动创建")

                        mode = gr.Radio(
                            ["普通单关键词模式", "关键词组模式", "自定义分类器模式"],
                            value="普通单关键词模式",
                            label="分类模式",
                        )
                        keywords_box = gr.Group(visible=True)
                        groups_box = gr.Group(visible=False)
                        custom_mode_box = gr.Group(visible=False)

                        with keywords_box:
                            keywords_str = gr.Textbox(
                                label="关键词 (逗号分隔)",
                                placeholder="shoes, bag, hat",
                            )
                        with groups_box:
                            groups_str = gr.Textbox(
                                label="关键词组 (每行一个)",
                                placeholder="face:face\nseqing:feet,shoes",
                                lines=5,
                            )
                            gr.Markdown(
                                "格式: `文件夹名:关键词1,关键词2`",
                                elem_classes="info-box",
                            )
                        with custom_mode_box:
                            custom_pkl = gr.Textbox(
                                label="📁 .pkl 模型文件路径",
                                placeholder="D:\\PY\\train\\classifier.pkl",
                            )
                            gr.Markdown(
                                "可选: 输入关键词先筛选，再用分类器细分\n"
                                "留空 = 对所有图片直接分类",
                                elem_classes="info-box",
                            )
                            custom_prefilter = gr.Textbox(
                                label="预筛选关键词 (可选，逗号分隔)",
                                placeholder="face  (留空则分类所有图片)",
                            )

                        model_name = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="CLIP ViT-L-14-336",
                            label="模型",
                        )
                        threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.001,
                            value=0.15,
                            label="阈值 (余弦相似度，范围 -1~1，建议 0.10~0.25)",
                        )

                        with gr.Row():
                            run_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
                            load_btn = gr.Button("📋 加载上次配置", size="lg")

                    with gr.Column(scale=2):
                        summary = gr.Textbox(label="结果摘要", lines=15, interactive=False)
                        result_table = gr.Dataframe(
                            headers=["文件名", "匹配关键词", "最高分", "最佳关键词", "自定义子类别"],
                            label="详细结果",
                            interactive=False,
                            wrap=True,
                        )

                def toggle_mode(m):
                    if m == "关键词组模式":
                        return gr.Group(visible=False), gr.Group(visible=True), gr.Group(visible=False)
                    elif m == "自定义分类器模式":
                        return gr.Group(visible=False), gr.Group(visible=False), gr.Group(visible=True)
                    return gr.Group(visible=True), gr.Group(visible=False), gr.Group(visible=False)

                mode.change(toggle_mode, mode, [keywords_box, groups_box, custom_mode_box])
                model_name.change(on_model_change, model_name, threshold)

                run_btn.click(
                    do_classify,
                    [src_dir, out_dir, mode, keywords_str, groups_str, model_name, threshold, custom_pkl, custom_prefilter],
                    [summary, result_table, out_dir],
                )
                load_btn.click(
                    do_load_last_config,
                    [],
                    [src_dir, out_dir, mode, keywords_str, model_name, threshold, summary],
                )

            # ── Tab 2: 模型自检 ──
            with gr.TabItem("🔬 模型自检"):
                with gr.Row():
                    with gr.Column(scale=1):
                        diag_model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="CLIP ViT-L-14-336",
                            label="选择模型",
                        )
                        diag_btn = gr.Button("🔬 开始自检", variant="primary")
                    with gr.Column(scale=2):
                        diag_result = gr.Textbox(label="自检结果", lines=25, interactive=False)

                diag_btn.click(do_diagnose, [diag_model], [diag_result])

            # ── Tab 3: 环境配置 ──
            with gr.TabItem("⚙️ 环境依赖"):
                env_btn = gr.Button("🔍 检查并安装依赖", variant="primary")
                env_result = gr.Textbox(label="环境检查结果", lines=25, interactive=False)
                env_btn.click(do_check_env, [], [env_result])

        # ── 底部: 退出按钮 ──
        gr.Markdown("---")
        with gr.Row():
            gr.Markdown("""
            **阈值说明：**
            - **CLIP**：余弦相似度，范围 -1~1。语义相似通常 0.2~0.4，建议从 0.15 开始
            - **SigLIP**：概率分布（总和=1）。多个关键词时每个的"随机水平"是 1/N，建议从 0.01 开始
            - SigLIP 自动计算 chance level × 1.2 作为有效阈值下限
            """)
            exit_btn = gr.Button("🚪 退出程序", variant="stop", size="lg", scale=0)

        exit_status = gr.Textbox(visible=False)
        exit_btn.click(fn=do_exit, outputs=[exit_status])

    return app


if __name__ == "__main__":
    # 信号处理：Ctrl+C 时清理退出
    def _signal_handler(sig, frame):
        print("\n正在退出...")
        do_exit()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    app = build_ui()
    app.launch(
        server_name="127.0.0.1",  # 仅本地访问
        server_port=7860,
        inbrowser=True,
        share=False,
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; font-size: 1.5em; margin-bottom: 0.5em; }
        .info-box { background: #e8f5e9; padding: 10px; border-radius: 8px; font-size: 0.9em; }
        """,
    )
