"""
自定义分类器训练工具
CLIP 特征提取 + SVM，128 张样本即可训练
独立于主程序，训练完输出 .pkl 模型文件

用法:
    训练: python custom_classifier_trainer.py --data D:\\data\\face
    UI:   python custom_classifier_trainer.py --ui
"""
import os
import sys
import pickle
import argparse
from pathlib import Path


def train_classifier(data_dir: str, output_path: str):
    """
    训练自定义二分类器（generator，yield 实时日志）

    data_dir 结构:
        data_dir/
            candid/   ← 正例
                img1.jpg
            normal/   ← 反例
                img1.jpg

    输出: yield log 行，最后 yield ("RESULT", model_data, output_path)
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import open_clip

    data_path = Path(data_dir)
    if not data_path.exists():
        yield f"❌ 数据目录不存在: {data_dir}"
        return

    # 扫描类别
    categories = sorted([
        d.name for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    if len(categories) < 2:
        yield f"❌ 至少需要 2 个类别文件夹，当前: {categories}"
        return

    yield "═══ 自定义分类器训练 ═══"
    yield f"  数据目录: {data_dir}"
    yield f"  类别: {categories}"

    # 收集图片
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    samples = []
    for idx, cat in enumerate(categories):
        cat_dir = data_path / cat
        imgs = [f for f in cat_dir.iterdir() if f.suffix.lower() in img_exts]
        if not imgs:
            yield f"❌ 类别 '{cat}' 中没有图片"
            return
        for img_path in imgs:
            samples.append((str(img_path), idx))
        yield f"  {cat}: {len(imgs)} 张"

    if len(samples) < 4:
        yield f"❌ 样本太少 ({len(samples)})，至少需要 4 张"
        return

    yield f"  总样本: {len(samples)}"

    # 加载 CLIP
    yield "\n  加载 CLIP ViT-L-14-336..."
    MODEL_NAME = "ViT-L-14-336"
    PRETRAINED = "openai"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()

    yield f"  设备: {device}"

    # 提取特征：批量处理 + 多线程加载
    yield "  提取特征中..."
    from concurrent.futures import ThreadPoolExecutor as _ThreadPool

    features = []
    labels = []
    errors = []
    batch_size = 256
    total = len(samples)

    def _load_one(item):
        path, label = item
        try:
            return preprocess(Image.open(path).convert("RGB")), label, None
        except Exception as e:
            return torch.zeros(3, 336, 336), label, f"{os.path.basename(path)}: {e}"

    with _ThreadPool(max_workers=8) as pool:
        for batch_start in range(0, total, batch_size):
            batch_items = samples[batch_start:batch_start + batch_size]
            loaded = list(pool.map(_load_one, batch_items))

            tensors = torch.stack([t for t, _, _ in loaded]).to(device, non_blocking=True)
            if device == "cuda":
                tensors = tensors.half()

            with torch.no_grad():
                feats = F.normalize(model.encode_image(tensors), dim=-1)

            feats_np = feats.float().cpu().numpy()
            for j, (_, label, err) in enumerate(loaded):
                if err:
                    errors.append(err)
                else:
                    features.append(feats_np[j])
                    labels.append(label)

            done = min(batch_start + batch_size, total)
            yield f"  {done}/{total}"

    features = np.array(features)
    labels = np.array(labels)
    yield f"  特征矩阵: {features.shape}"

    if len(features) < 4:
        yield f"❌ 有效样本太少 ({len(features)})"
        return

    # 训练 SVM
    yield "  训练 SVM..."
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    svm = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale")
    svm.fit(features_scaled, labels)

    train_acc = svm.score(features_scaled, labels)
    yield f"  训练准确率: {train_acc:.2%}"

    # 保存
    model_data = {
        "categories": list(categories),
        "scaler": scaler,
        "svm": svm,
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "feature_dim": int(features.shape[1]),
        "num_samples": len(samples),
        "train_acc": float(train_acc),
        "errors": errors,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    size_kb = os.path.getsize(output_path) / 1024
    yield f"\n✅ 模型已保存: {output_path}"
    yield f"  类别: {categories}"
    yield f"  准确率: {train_acc:.2%}"
    yield f"  文件大小: {size_kb:.1f} KB"

    if errors:
        yield f"\n  ⚠ 跳过了 {len(errors)} 张有问题的图片"

    # 清理
    if device == "cuda":
        del model
        torch.cuda.empty_cache()

    yield ("RESULT", model_data, output_path)


def predict_batch(image_paths: list, model_path: str, progress=None):
    """
    批量预测：用训练好的分类器判断每张图片属于哪个子类别
    批量推理 + 多线程加载，性能与主程序一致

    返回: list of dict
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    import open_clip
    from concurrent.futures import ThreadPoolExecutor as _ThreadPool

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    categories = model_data["categories"]
    scaler = model_data["scaler"]
    svm = model_data["svm"]
    model_name = model_data["model_name"]
    pretrained = model_data["pretrained"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()

    def _load_one(path):
        try:
            return preprocess(Image.open(path).convert("RGB"))
        except Exception:
            return torch.zeros(3, 336, 336)

    results = []
    batch_size = 256
    total = len(image_paths)

    with _ThreadPool(max_workers=8) as pool:
        for batch_start in range(0, total, batch_size):
            batch_paths = image_paths[batch_start:batch_start + batch_size]
            tensors_list = list(pool.map(_load_one, batch_paths))
            tensors = torch.stack(tensors_list).to(device, non_blocking=True)
            if device == "cuda":
                tensors = tensors.half()

            with torch.no_grad():
                feats = F.normalize(model.encode_image(tensors), dim=-1)

            feats_np = feats.float().cpu().numpy()
            feats_scaled = scaler.transform(feats_np)
            probs_all = svm.predict_proba(feats_scaled)

            for j, path in enumerate(batch_paths):
                probs = probs_all[j]
                best_idx = int(np.argmax(probs))
                results.append({
                    "path": path,
                    "category": categories[best_idx],
                    "confidence": float(probs[best_idx]),
                    "probs": {cat: float(p) for cat, p in zip(categories, probs)},
                })

            done = min(batch_start + batch_size, total)
            if progress:
                progress(done / total, desc=f"自定义分类 {done}/{total}")

    if device == "cuda":
        del model
        torch.cuda.empty_cache()

    return results


# ── 独立 Gradio 训练界面 ──
def build_trainer_ui():
    import gradio as gr

    def scan_data(data_dir):
        """扫描数据目录，返回类别和图片数量"""
        if not data_dir or not os.path.isdir(data_dir):
            return "❌ 请选择有效的数据目录"
        img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        categories = sorted([
            d.name for d in Path(data_dir).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        if not categories:
            return "❌ 未找到子文件夹（每个子文件夹 = 一个类别）"
        lines = ["═══ 数据预览 ═══"]
        total = 0
        for cat in categories:
            cat_dir = Path(data_dir) / cat
            imgs = [f for f in cat_dir.iterdir() if f.suffix.lower() in img_exts]
            count = len(imgs)
            total += count
            status = "✓" if count >= 10 else "⚠ 太少"
            lines.append(f"  {cat}: {count} 张 {status}")
        lines.append(f"\n  总计: {total} 张, {len(categories)} 个类别")
        if total < 4:
            lines.append("  ❌ 至少需要 4 张图片")
        elif total < 20:
            lines.append("  ⚠ 样本较少，建议每类 30+ 张以获得更好效果")
        else:
            lines.append("  ✓ 样本量足够")
        return "\n".join(lines)

    def do_train(data_dir, output_dir):
        if not data_dir or not os.path.isdir(data_dir):
            yield "❌ 请选择有效的数据目录"
            return
        # 输出路径
        if output_dir and os.path.isdir(output_dir):
            output_path = os.path.join(output_dir, "classifier.pkl")
        else:
            output_path = os.path.join(data_dir, "classifier.pkl")

        try:
            log = ""
            model_path = ""
            for line in train_classifier(data_dir, output_path):
                if isinstance(line, tuple) and line[0] == "RESULT":
                    _, model_data, path = line
                    model_path = path
                else:
                    log += line + "\n"
                    yield log
        except Exception as e:
            import traceback
            yield f"❌ 错误: {e}\n{traceback.format_exc()}"

    def do_test(model_path, test_image):
        """用训练好的模型测试单张图片"""
        if not model_path or not os.path.isfile(model_path):
            return "❌ 请先训练模型或选择 .pkl 文件"
        if not test_image:
            return "❌ 请选择一张测试图片"
        try:
            from custom_classifier_trainer import predict_batch
            results = predict_batch([test_image], model_path)
            if not results:
                return "❌ 预测失败"
            r = results[0]
            if "error" in r:
                return f"❌ {r['error']}"
            lines = [
                "═══ 测试结果 ═══",
                f"  图片: {os.path.basename(test_image)}",
                f"  预测: {r['category']}",
                f"  置信度: {r['confidence']:.2%}",
                "",
                "  概率分布:",
            ]
            for cat, prob in sorted(r["probs"].items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                lines.append(f"    {cat}: {prob:.2%} {bar}")
            return "\n".join(lines)
        except Exception as e:
            import traceback
            return f"❌ 错误: {e}\n{traceback.format_exc()}"

    def pick_data_folder():
        """弹出系统文件夹选择框"""
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="选择数据目录")
        root.destroy()
        return folder if folder else ""

    def pick_output_folder():
        """弹出系统文件夹选择框"""
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="选择输出目录")
        root.destroy()
        return folder if folder else ""

    with gr.Blocks(title="自定义分类器训练") as app:
        gr.Markdown("# 🎯 自定义分类器训练工具\n"
                     "CLIP 特征提取 + SVM 训练，每类建议 30~100 张图片")

        with gr.Tabs():
            # ── Tab 1: 训练 ──
            with gr.TabItem("🚀 训练"):
                gr.Markdown("**数据目录结构:**\n"
                             "```\n"
                             "your_data/\n"
                             "  candid/   ← 正例（你想识别的目标）\n"
                             "    img1.jpg\n"
                             "  normal/   ← 反例（对比组）\n"
                             "    img1.jpg\n"
                             "```")

                with gr.Row():
                    data_dir = gr.Textbox(label="📁 数据目录路径", placeholder="点击右侧按钮选择文件夹")
                    pick_data_btn = gr.Button("📂 选择", scale=0)
                with gr.Row():
                    output_dir = gr.Textbox(label="📂 输出目录路径 (留空则保存在数据目录下)", placeholder="点击右侧按钮选择文件夹")
                    pick_out_btn = gr.Button("📂 选择", scale=0)

                pick_data_btn.click(fn=pick_data_folder, outputs=[data_dir])
                pick_out_btn.click(fn=pick_output_folder, outputs=[output_dir])

                preview_btn = gr.Button("👁️ 预览数据")
                preview_box = gr.Textbox(label="数据预览", lines=8, interactive=False)

                train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
                result_box = gr.Textbox(label="训练日志", lines=15, interactive=False)

                preview_btn.click(fn=scan_data, inputs=[data_dir], outputs=[preview_box])
                train_btn.click(
                    fn=do_train,
                    inputs=[data_dir, output_dir],
                    outputs=[result_box],
                )

            # ── Tab 2: 测试 ──
            with gr.TabItem("🧪 测试"):
                gr.Markdown("选择一张图片，用训练好的模型预测")

                with gr.Row():
                    test_model = gr.File(
                        label="选择 .pkl 模型文件",
                        file_types=[".pkl"],
                        type="filepath",
                    )
                    test_image = gr.File(
                        label="选择测试图片",
                        file_types=[".jpg", ".jpeg", ".png", ".webp"],
                        type="filepath",
                    )

                test_btn = gr.Button("🧪 开始测试", variant="primary")
                test_result = gr.Textbox(label="测试结果", lines=10, interactive=False)

                test_btn.click(fn=do_test, inputs=[test_model, test_image], outputs=[test_result])

    return app


def _kill_port(port):
    """杀掉占用指定端口的进程"""
    import subprocess
    try:
        r = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        for line in r.stdout.split("\n"):
            if f":{port}" in line and "LISTEN" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit():
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True)
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自定义分类器训练工具")
    parser.add_argument("--data", "-d", help="数据目录路径")
    parser.add_argument("--output", "-o", help="输出 .pkl 路径")
    parser.add_argument("--ui", action="store_true", help="启动 Gradio 训练界面 (端口 7861)")
    args = parser.parse_args()

    if args.ui:
        _kill_port(7861)
        app = build_trainer_ui()
        app.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
    elif args.data:
        output = args.output or os.path.join(args.data, "classifier.pkl")
        _, msg = train_classifier(args.data, output)
        print(msg)
    else:
        print("用法:")
        print("  命令行训练: python custom_classifier_trainer.py --data D:\\data\\face")
        print("  启动 UI:   python custom_classifier_trainer.py --ui")
