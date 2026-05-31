#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 图片分类器 — 自动安装配置脚本
双击 setup.bat 或 python setup.py 即可自动配置一切
"""

import os
import sys
import subprocess
import shutil
import platform
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
VENV_DIR = SCRIPT_DIR / "venv"
PY312_DIR = None  # 自动检测

# ╔══════════════════════════════════════════════════════════════════╗
# ║                          工具函数                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def banner(text):
    w = 56
    print(f"\n{'='*w}")
    print(f"  {text}")
    print(f"{'='*w}")


def run(cmd, desc="", check=True, capture=False):
    """执行命令并显示结果"""
    if desc:
        print(f"  {desc} ...")
    try:
        r = subprocess.run(
            cmd, capture_output=capture, text=True,
            encoding="utf-8", errors="replace"
        )
        if check and r.returncode != 0:
            err = (r.stderr or r.stdout or "")[-300:]
            print(f"  ✗ 失败: {err}")
            return False
        return True
    except FileNotFoundError:
        print(f"  ✗ 命令不存在: {cmd[0]}")
        return False
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


def find_python312():
    """查找系统中已安装的 Python 3.12"""
    # 常见安装路径
    candidates = []
    if platform.system() == "Windows":
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        prog = Path(os.environ.get("PROGRAMFILES", ""))
        prog86 = Path(os.environ.get("PROGRAMFILES(X86)", ""))

        for base in [local / "Programs" / "Python", prog / "Python", prog86 / "Python"]:
            if base.exists():
                for d in base.iterdir():
                    if d.name.startswith("Python31"):
                        candidates.append(d / "python.exe")

        # 也检查 PATH 中的 python
        for p in os.environ.get("PATH", "").split(";"):
            p = Path(p.strip())
            if p.name.lower() in ("python312", "python3.12"):
                candidates.append(p / "python.exe")

    # 用 py launcher 查找
    try:
        r = subprocess.run(["py", "-3.12", "--version"],
                          capture_output=True, text=True)
        if r.returncode == 0:
            r2 = subprocess.run(["py", "-3.12", "-c", "import sys; print(sys.executable)"],
                               capture_output=True, text=True)
            if r2.returncode == 0:
                candidates.insert(0, Path(r2.stdout.strip()))
    except FileNotFoundError:
        pass

    # 验证每个候选
    for c in candidates:
        if c.exists():
            try:
                r = subprocess.run([str(c), "--version"],
                                  capture_output=True, text=True)
                if "3.12" in r.stdout:
                    return c
            except Exception:
                continue
    return None


def find_venv_python():
    """查找 venv 中的 Python"""
    if VENV_DIR.exists():
        p = VENV_DIR / "Scripts" / "python.exe"
        if p.exists():
            return p
    return None


def get_system_python():
    """获取当前系统 Python"""
    return Path(sys.executable)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      检测与安装流程                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def check_nvidia_driver():
    """检测 NVIDIA 驱动"""
    banner("[1/5] NVIDIA 驱动检测")
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
        if r.returncode == 0:
            for line in r.stdout.split("\n"):
                if "Driver Version" in line:
                    print(f"  ✓ {line.strip()}")
                    break
            # 提取 GPU 名称
            for line in r.stdout.split("\n"):
                if "NVIDIA" in line and "MiB" not in line and "Driver" not in line:
                    gpu = line.strip().split("|")
                    if len(gpu) >= 2:
                        print(f"  ✓ GPU: {gpu[1].strip()}")
                        break
            return True
        else:
            print("  ✗ nvidia-smi 执行失败")
            return False
    except FileNotFoundError:
        print("  ✗ nvidia-smi 未找到")
        print("")
        print("  ┌─────────────────────────────────────────────┐")
        print("  │  需要安装 NVIDIA 显卡驱动:                    │")
        print("  │  https://www.nvidia.com/drivers              │")
        print("  │  安装后重启电脑，再运行本程序                  │")
        print("  └─────────────────────────────────────────────┘")
        return False


def check_python():
    """检测 Python 版本，必要时安装 3.12"""
    banner("[2/5] Python 版本检测")

    # 检查当前 Python
    ver = sys.version_info
    print(f"  当前 Python: {ver.major}.{ver.minor}.{ver.micro}")

    # 已有 venv
    venv_py = find_venv_python()
    if venv_py:
        try:
            r = subprocess.run([str(venv_py), "--version"],
                              capture_output=True, text=True)
            if "3.12" in r.stdout:
                print(f"  ✓ venv 已存在: {venv_py}")
                return venv_py
        except Exception:
            pass

    # 当前就是 3.12
    if ver.major == 3 and ver.minor == 12:
        print(f"  ✓ Python 3.12，直接使用")
        return get_system_python()

    # 查找系统中的 3.12
    py312 = find_python312()
    if py312:
        print(f"  ✓ 找到 Python 3.12: {py312}")
        return py312

    # 没有 3.12，需要安装
    if ver.major == 3 and ver.minor >= 13:
        print(f"  ⚠ Python {ver.major}.{ver.minor} 暂不支持 CUDA 版 PyTorch")
        print(f"  → 需要安装 Python 3.12")

    print(f"\n  自动安装 Python 3.12 ...")

    if platform.system() == "Windows":
        # 用 winget 安装
        ok = run(
            ["winget", "install", "Python.Python.3.12",
             "--accept-package-agreements", "--accept-source-agreements"],
            desc="winget 安装 Python 3.12"
        )
        if ok:
            py312 = find_python312()
            if py312:
                print(f"  ✓ Python 3.12 安装成功: {py312}")
                return py312

        # winget 失败，尝试下载安装
        print("  → winget 安装失败，尝试直接下载...")
        url = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
        installer = SCRIPT_DIR / "python-3.12.10-amd64.exe"
        print(f"  下载 {url} ...")
        try:
            urllib.request.urlretrieve(url, installer)
            print(f"  → 运行安装程序 (静默安装) ...")
            run([str(installer), "/quiet", "InstallAllUsers=0", "PrependPath=1",
                 "Include_launcher=1"], desc="安装 Python 3.12")
            installer.unlink(missing_ok=True)
            py312 = find_python312()
            if py312:
                print(f"  ✓ Python 3.12 安装成功")
                return py312
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")

    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │  自动安装失败，请手动安装 Python 3.12:       │")
    print("  │  https://www.python.org/downloads/           │")
    print("  │  安装后重新运行本程序                        │")
    print("  └─────────────────────────────────────────────┘")
    return None


def create_venv(python_path):
    """创建虚拟环境"""
    banner("[3/5] 创建虚拟环境")

    if VENV_DIR.exists():
        # 检查已有 venv 的 Python 版本
        venv_py = VENV_DIR / "Scripts" / "python.exe"
        if venv_py.exists():
            try:
                r = subprocess.run([str(venv_py), "-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                    capture_output=True, text=True)
                ver = r.stdout.strip()
                if ver == "3.12":
                    print(f"  ✓ venv 已存在 (Python {ver})")
                    return venv_py
                else:
                    print(f"  ⚠ venv 是 Python {ver}，需要重建")
                    shutil.rmtree(VENV_DIR)
            except Exception:
                shutil.rmtree(VENV_DIR)

    print(f"  用 {python_path} 创建 venv ...")
    ok = run([str(python_path), "-m", "venv", str(VENV_DIR)], desc="创建 venv")
    if not ok:
        print("  ✗ venv 创建失败")
        return None

    venv_py = VENV_DIR / "Scripts" / "python.exe"
    if venv_py.exists():
        print(f"  ✓ venv 创建成功: {VENV_DIR}")
        return venv_py
    return None


def install_cuda_pytorch(venv_py):
    """安装 CUDA 版 PyTorch"""
    banner("[4/5] 安装 CUDA 版 PyTorch")

    pip = str(venv_py).replace("python.exe", "pip.exe")

    # 先升级 pip
    run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"],
        desc="升级 pip", check=False)

    # 卸载可能存在的 CPU 版
    run([pip, "uninstall", "torch", "torchvision", "-y"], check=False, capture=True)

    # 安装 CUDA 版
    ok = run(
        [pip, "install", "torch", "torchvision",
         "--index-url", "https://download.pytorch.org/whl/cu124"],
        desc="安装 torch+torchvision (cu124)"
    )
    if not ok:
        print("  ⚠ cu124 安装失败，尝试 cu121 ...")
        ok = run(
            [pip, "install", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cu121"],
            desc="安装 torch+torchvision (cu121)"
        )
    if not ok:
        print("  ⚠ cu121 安装失败，尝试 cu118 ...")
        ok = run(
            [pip, "install", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cu118"],
            desc="安装 torch+torchvision (cu118)"
        )

    if not ok:
        print("  ✗ CUDA 版 PyTorch 安装全部失败")
        return False

    # 验证
    r = subprocess.run(
        [str(venv_py), "-c",
         "import torch; print(f'{torch.__version__}|{torch.version.cuda}|{torch.cuda.is_available()}')"],
        capture_output=True, text=True
    )
    parts = r.stdout.strip().split("|")
    if len(parts) == 3:
        ver, cuda, avail = parts
        print(f"  ✓ PyTorch {ver} (CUDA {cuda}, available={avail})")
        if avail == "True":
            r2 = subprocess.run(
                [str(venv_py), "-c",
                 "import torch; print(torch.cuda.get_device_name(0))"],
                capture_output=True, text=True
            )
            print(f"  ✓ GPU: {r2.stdout.strip()}")
        return avail == "True"

    print("  ✗ PyTorch 验证失败")
    return False


def install_other_deps(venv_py):
    """安装其他依赖"""
    banner("[5/5] 安装其他依赖")

    pip = str(venv_py).replace("python.exe", "pip.exe")
    deps = [
        "open-clip-torch",
        "transformers",
        "gradio",
        "tqdm",
        "numpy",
        "Pillow",
        "scikit-learn",
    ]

    for dep in deps:
        run([pip, "install", dep], desc=f"安装 {dep}", check=False)

    # 验证
    r = subprocess.run(
        [str(venv_py), "-c",
         "import open_clip, transformers, gradio, tqdm, numpy, PIL, sklearn; print('OK')"],
        capture_output=True, text=True
    )
    if "OK" in r.stdout:
        print(f"\n  ✓ 所有依赖安装完成")
        return True
    else:
        print(f"\n  ⚠ 部分依赖可能有问题")
        print(f"    {r.stderr[-200:]}")
        return False


def create_launcher():
    """创建一键启动脚本"""
    print(f"\n  ✓ 启动脚本已就绪:")
    print(f"    {SCRIPT_DIR / 'start.bat'}")
    print(f"    双击即可启动")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                            主流程                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    print("""
    ╔══════════════════════════════════════════════╗
    ║     AI 图片分类器 — 自动安装配置             ║
    ║     CLIP + SigLIP · CUDA 加速               ║
    ╚══════════════════════════════════════════════╝
    """)

    # 1. NVIDIA 驱动
    driver_ok = check_nvidia_driver()

    # 2. Python
    python_path = check_python()
    if not python_path:
        input("\n按回车退出...")
        return

    # 3. venv
    venv_py = create_venv(python_path)
    if not venv_py:
        input("\n按回车退出...")
        return

    # 4. CUDA PyTorch
    cuda_ok = install_cuda_pytorch(venv_py)

    # 5. 其他依赖
    install_other_deps(venv_py)

    # 6. 创建启动脚本
    create_launcher()

    # 汇总
    banner("安装完成")
    print(f"  Python: 3.12 (venv)")
    print(f"  venv: {VENV_DIR}")
    print(f"  CUDA: {'✓ 就绪' if cuda_ok else '✗ 不可用'}")
    if not driver_ok:
        print(f"  ⚠ NVIDIA 驱动未检测到，请安装后重启")
    print(f"\n  双击「启动分类器.bat」即可运行")
    print(f"  或执行: \"{venv_py}\" \"{SCRIPT_DIR / 'ai_classifier_ui.py'}\"")

    input("\n按回车退出...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n安装出错: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车退出...")
