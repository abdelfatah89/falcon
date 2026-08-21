from transformers import pipeline


def main():
    generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

    result = generator("The future of AI is")
    print(result)


if __name__ == "__main__":
    main()
