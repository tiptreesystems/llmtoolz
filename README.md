# LLMToolz

## Usage

```bash
pip install llmtoolz
```

```bash
export OPENAI_KEY="your-api-key"
```


## Bot

`Bot` is the fundamental building block of the `llmtoolz` library. They are used to interact with an LLM service provider's API (e.g., TogetherCompute, OpenAI). They maintain a history of user and system events, truncate long conversation histories, and generally manage the unreliability of the API (e.g., rate limits, timeouts, etc.). 

```python
from llmtoolz import Bot
bot = Bot(
    name="joey_the_assistant",
    model_name="gpt-4",
    system_prompt="You are a helpful assistant named {assistant_name}.",
    system_prompt_format_kwargs={"assistant_name": "Joey"},
)

bot.history.add_user_event(user_prompt="I need help with something.")

response = bot.complete(max_tokens=1024).content

print(response)

# Output: "Hi, I'm Joey, what do you need help with?"
```


## Bjork

`bjork` is a wrapper for a web agent built using `Bot`s. It can use tools including a `search_tool`, `web_page_query_tool`, and a `people_lookup_tool`. 


```python
from llmtoolz import bjork
recipe = bjork(query="How to make a cake", validate_query=True)

print(recipe)
# Output: "Here's a recipe for a cake... [bjorkycookbook](https://www.bjorkycookbook.com/recipe/a_digital_cake)"
```




## Lorimer

