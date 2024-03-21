from llmtoolz import Bot


def example_1():
    bot = Bot(
        name="assistant",
        model_name="gpt-4",
        system_prompt="You are a helpful assistant named {assistant_name}.",
        system_prompt_format_kwargs={"assistant_name": "Joey"},
    )

    user_prompt = "I need help with something."
    bot.history.add_user_event(user_prompt)

    response = bot.complete(max_tokens=1024).content

    print(response)

    # Output: "Hi, I'm Joey, what do you need help with?"


def main():
    example_1()


if __name__ == "__main__":
    main()
