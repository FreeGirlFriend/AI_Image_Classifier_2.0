#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
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
