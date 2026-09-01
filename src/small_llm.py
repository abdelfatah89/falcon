import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from typing import List


class FalconLLM:
    def __init__(self, model_path: str | None = None):
        model_path = model_path or os.environ.get(
            "FALCON_MODEL_PATH",
            "tiiuae/Falcon-H1-1.5B-Instruct",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            #trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            #trust_remote_code=True,
        )
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True
        )
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(
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

    def generate(self, prompt, max_token):
        return self.generate(prompt, max_token=max_token)
