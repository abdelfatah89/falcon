import torch

from src.small_llm import FalconLLM
from src.llm_model import LLModel


def main():
    model = FalconLLM()
    prompt = "What is the capital of Morocco?"

    tokens_ids = model.encode(prompt)
    print("Tokens IDs:", tokens_ids)

    logits = model.get_logits(tokens_ids)
    #print(logits)

    last_logits = logits[0, -1]
    print(logits.shape)

    probabilities = torch.softmax(last_logits, dim=-1)

    top_probs, top_token_ids = torch.topk(probabilities, 5)

    print("\nNext-token candidates:")

    for probability, token_id in zip(top_probs, top_token_ids):
        token = model.decode([token_id.item()])
        
        print(
            f"Token ID: {token_id.item():6d} | "
            f"Token: {token!r:15} | "
            f"Probability: {probability.item():.2%}"
        )
    

    


if __name__ == "__main__":
    main()
