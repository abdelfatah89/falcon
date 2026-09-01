from typing import List, Dict, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline)


class LLModel:
    def __init__(self,
                 model_name: str = r"D:\AI_models\Falcon-H1-1.5B-Instruct",
                 *,
                 dtype: torch.dtype | None = None,
                 trust_remote_code: bool = True
                 ) -> None:

        self._model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        model_kwargs = {
            "torch_dtype": self._dtype,
            "trust_remote_code": trust_remote_code,
        }
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if self.device.type != "cuda":
            self._model.to(self.device)
        self._model.eval()

        self.generator = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response from the LLM based on the given prompt.
        """
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(prompt, str):
            raise TypeError("Expected string prompt from apply_chat_template")

        result = self.generator(
            prompt,
            do_sample=False,
            return_full_text=False,
        )

        return cast(str, result[0]["generated_text"])

    def generate_prompt(self,
                        question: str,
                        sources: List[str] | None = None
                        ) -> List[Dict[str, str]]:
        """
        Generate a prompt for the LLM based on
        the question and retrieved sources.
        """
        sources = sources or []
        context = "\n\n".join(
                    f"[Context {i + 1}]\n{source}"
                    for i, source in enumerate(sources)
                )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. Answer the user's questions"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ]

        return messages
