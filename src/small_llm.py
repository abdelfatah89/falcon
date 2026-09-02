import os
import json
from typing import List

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def save_file(file_path, content):
    with open(file_path, "w") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)


class FalconLLM:
    def __init__(self, model_path: str | None = None):
        self.model_name = model_path or os.environ.get(
            "FALCON_MODEL_PATH",
            "tiiuae/Falcon-H1-1.5B-Instruct",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self._tokenizer)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> List[int]:
        tokens = self._tokenizer.encode(
            text,
            add_special_tokens=True
        )
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @torch.no_grad()
    def get_logits(self, tokens: List[int]) -> torch.Tensor:
        input_ids = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=self.device
        )
        output = self.model(input_ids=input_ids)
        return output[0]

    def get_tokenizer_path(self) -> str:
        tokenizer_name = self._tokenizer.vocab_files_names.get('tokenizer_file', "tokenizer.json")
        if tokenizer_name not in os.listdir("Falcon-H1-1.5B-Instruct"):
            print("Downloading tokenizer.json from Hugging Face Hub...")
            path = hf_hub_download(
                repo_id="tiiuae/Falcon-H1-1.5B-Instruct",
                filename="tokenizer.json",
                local_dir="./Falcon-H1-1.5B-Instruct",
            )
            return path
        return os.path.join("Falcon-H1-1.5B-Instruct", tokenizer_name)

    def get_vocab_path(self) -> str:
        vocab_path = "Falcon-H1-1.5B-Instruct/vocab.json"
        tokenizer_path = self.get_tokenizer_path()
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        vocab = tokenizer_data.get("model", {}).get("vocab", {})
        with open(vocab_path, "w", encoding="utf-8") as f:
            save_file(vocab_path, vocab)

        return vocab_path

    def get_merges_path(self) -> str:
        merges_path = "Falcon-H1-1.5B-Instruct/merges.txt"
        tokenizer_path = self.get_tokenizer_path()
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        merges = tokenizer_data.get("model", {}).get("merges", [])
        with open(merges_path, "w", encoding="utf-8") as f:
            save_file(merges_path, merges)

        return merges_path

    def generate(self, prompt, max_token):
        return self.generate(prompt, max_token=max_token)
