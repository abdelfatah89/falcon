import torch

from src.small_llm import FalconLLM
from src.llm_model import LLModel


def main():
    model = FalconLLM()
    prompt = "What is the capital of Morocco?"

    #tokens_ids = model.encode(prompt)
    #print("Tokens IDs:", tokens_ids)

    #logits = model.get_logits(tokens_ids)
    ##print(logits)

    #last_logits = logits[0, -1]
    #print(logits.shape)

    #probabilities = torch.softmax(last_logits, dim=-1)

    #top_probs, top_token_ids = torch.topk(probabilities, 5)

    #print("\nNext-token candidates:")

    #for probability, token_id in zip(top_probs, top_token_ids):
    #    token = model.decode([token_id.item()])
        
    #    print(
    #        f"Token ID: {token_id.item():6d} | "
    #        f"Token: {token!r:15} | "
    #        f"Probability: {probability.item():.2%}"
        #)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="tiiuae/Falcon-H1-1.5B-Instruct",
        filename="tokenizer.json",
        local_dir="./Falcon-H1-1.5B-Instruct",
    )

    import json
    vocab ={}
    with open("Falcon-H1-1.5B-Instruct/tokenizer.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(vocab.keys())
    print(vocab["model"])
    print(vocab["model"].keys())
    print(len(vocab["model"]["vocab"]))

    


if __name__ == "__main__":
    main()
