#!/usr/bin/env python3

import os


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT_DIR = get_root_dir()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
