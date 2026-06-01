#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 图片分类器 — 启动入口
解决 Windows CMD GBK 编码乱码问题
"""

import sys
import os

# ═══════════════════════════════════════════════════════════════════
# 修复 Windows CMD 编码 GBK 乱码
# 用 reconfigure() 原地修改，不替换对象，避免破坏 isatty() 和 logging
# ═══════════════════════════════════════════════════════════════════
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from ai_classifier_ui import build_ui, do_exit

# Ctrl+C clean exit
signal.signal(signal.SIGINT, lambda *_: do_exit())
signal.signal(signal.SIGTERM, lambda *_: do_exit())

app = build_ui()
app.launch(
    server_name="127.0.0.1",
    server_port=7860,
    inbrowser=True,
    share=False,
)
