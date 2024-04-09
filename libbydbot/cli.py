import fire
from libbydbot.brain import embed


def main(question="What is a llama?"):
    response = embed.generate_response(question)
    print(response)

if __name__ == '__main__':
    fire.Fire(main)

