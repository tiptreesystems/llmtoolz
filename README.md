# LLMToolz
A set of utility functions for building LLM applications.

## Usage

```bash
pip install llmtoolz
```

```bash
export OPENAI_KEY="your-api-key"
export APIFY_KEY="your-apify-key"
```

## Bot

`Bot` is the fundamental building block of the `llmtoolz` library. They are used to interact with an LLM service
provider's API (e.g., TogetherCompute, OpenAI). They maintain a history of user and system events, truncate long
conversation histories, and generally manage the unreliability of the API (e.g., rate limits, timeouts, etc.).

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

`bjork` is a wrapper for a web agent built using `Bot`s. It can use tools including
a `search_tool`, `web_page_query_tool`, and a `people_lookup_tool`.

```python
from llmtoolz import bjork

recipe = bjork(query="How to make a cake", validate_query=True)

print(recipe)
# Output: "Here's a recipe for a cake... [bjorkycookbook](https://www.bjorkycookbook.com/recipe/a_digital_cake)"
```

## Lorimer

Lorimer is a recursive long-form document generator. Lorimer is an agent that produces the title and abstract of
sections and sub-sections of a document. Arnar is an agent that manages the creation of a specific section. Arnar uses
bjork to do research for that specific subsection, and may generate new subsections.

```python
from llmtoolz import lorimer

root_section = lorimer("Write a survey on GFlowNets")

root_section.persist(get_path("survey_on_gflownets.md"))

markdown_document = root_section.render()
print(markdown_document)
```


