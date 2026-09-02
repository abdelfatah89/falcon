import json
import os

import torch
from huggingface_hub import hf_hub_download

from src.small_llm import FalconLLM
from src.llm_model import LLModel


def save_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)


def main():
    _model = FalconLLM(r"/home/alaktaou/OSAKA/models/Falcon-H1-1.5B-Instruct")
    # _prompt = "What is the capital of Morocco?"
    print(_model.get_merges_path())


if __name__ == "__main__":
    main()
