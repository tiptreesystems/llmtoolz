from typing import Optional, List

import pandas as pd
from common.logger import logger

from .arnar_utils import ResearchAssistant
from .bot import Bot
from .multitool import find_tool, ToolBelt
from .utils import (
    find_markdown_block,
    DEFAULT_FLAGSHIP_MODEL,
    DEFAULT_FLAGSHIP_LONG_CONTEXT_MODEL,
    find_text_under_header,
)

from .scrape import WebPage


RESPONSE_INSTRUCTIONS = """# Instructions for Formulating your Final Response

It is now time to formulate a **comprehensive final response** based on the outline and research. It's often a good idea to (a) include direct quotes, (b) detail the specifics about the people and companies involved such as their names, titles, and locations, (c) include technical details such as dates, numbers, and statistics. 

Here are some ground rules for your final response: 

1. Your final response should be clear, and present important facts that you have learned during your research.
2. Your response should be comprehensive and not miss important facts. You may use arbitrary markdown syntax to structure your response.
3. Your answer should be neutral, objective and not opinionated. Remember, as a seasoned research AI, it is not your job to pass judgement (positive or negative), but to present the facts. DO NOT USE FLUFF AND FILLER CONTENT. 
4. Very importantly, **cite your sources using footnotes**. The research assistant will provide you with the sources it used to answer your questions.

Your response should go under the markdown header `# Final Response`. Your answer itself must be a Markdown codeblock, where you should use rich Markdown syntax:

```markdown
# Title 1
Your answer goes here [^1^]. You can include any markdown syntax here that you like [^2^].

# Title 2
More content may go here [^3^]. In addition, 
- You may include bullet points like this.
- And this.
...

You can also include numbered lists like this:
1. First item
2. Second item
...

# Sources
[^1^] [Title of source 1.](link)
[^2^] [Title of source 2.](link)
[^3^] [Title of source 3.](link)
```

**Great researchers cite sources!** 

You can do this, we're counting on you.
"""


def arnar(
    question: str,
    *,
    context: Optional[str] = None,
    max_num_tool_calls: int = 3,
    max_num_steps: int = 10,
    model_name: str = DEFAULT_FLAGSHIP_MODEL,
    bjork_kwargs: Optional[dict] = None,
    considered_web_pages_: Optional[List["WebPage"]] = None,
    document_research_question: Optional[str] = None,
    max_num_bjork_calls: int = 3,
) -> str:
    bot = Bot(name="arnar", model_name=model_name)

    tool_belt = ToolBelt(
        [
            ResearchAssistant(
                model_name=model_name,
                bjork_kwargs=bjork_kwargs,
                considered_web_pages_=considered_web_pages_,
            )
        ]
    )

    if document_research_question is not None:
        user_message = f"# Document-level Research Question\n\n{document_research_question}\n\n# Section Research Question\n\n{question}"
    else:
        user_message = f"# Section-level Research Question\n\n{question}"

    if context is not None:
        user_message = f"{user_message}\n\n# Context for Question\n\n{context}"

    # Add today's date
    date_str = pd.to_datetime("today").strftime("%B %d, %Y")
    user_message = f"{user_message}\n\n# Today's Date\n\n{date_str}"

    constraints = f"- You may use your tools a maximum of {max_num_tool_calls} times.\n"
    user_message = f"{user_message}\n\n# Constraints for Your Research\n\n{constraints}"

    bot.history.add_user_event(user_message)
    logger.debug(user_message)

    tool_call_count = 0
    step_count = 0
    steps_since_last_tool_call = -1
    final_response = None

    while True:
        response = bot.complete(max_tokens=1024, temperature=0.2)
        response = response["content"]
        logger.debug(response)

        if "# Tool Query" in response:
            if tool_call_count >= max_num_tool_calls:
                error_message = (
                    "# System Message\n\n"
                    "Error: you ran out of queries. Please output READY TO RESPOND now. "
                )
                bot.history.add_user_event(error_message)
            else:
                tool_search_query = find_text_under_header(response, "# Tool Query")

                found_tools = find_tool(
                    task=tool_search_query,
                    tools=tool_belt,
                    model_name=model_name,
                )
                if len(found_tools) > 0:
                    lines = ["# Found Tools", ""]
                    for tool in found_tools:
                        lines.append(tool.render())
                        lines.append("")
                    tool_lines = "\n".join(lines)
                    logger.debug(tool_lines)
                    bot.history.add_user_event(tool_lines)

        # Check if a tool is being used
        matching_tools = tool_belt.find_matching_tools(response)

        if len(matching_tools) == 1:
            steps_since_last_tool_call = -1
            tool = matching_tools[0]
            valid, message = tool.check_call_arguments(response)
            if not valid:
                error_message = (
                    f"Error calling tool {tool.tool_name}.{tool.function_name} "
                    f"-- {message}"
                )
                bot.history.add_user_event(error_message)
            else:
                answer = tool.use(response)
                formatted_answer = tool.format_result(answer)
                formatted_answer = (
                    f"{formatted_answer}\n\n"
                    f"# System Information\n\n"
                    f"Please think deeply and critically about the result under the markdown header '# Thoughts'. If there are inconsistencies or issues, please flag them here."
                )
                # Log and add to history
                logger.debug(formatted_answer)
                bot.history.add_user_event(formatted_answer)
                tool_call_count += 1

        elif len(matching_tools) > 1:
            steps_since_last_tool_call = -1
            # Scold the model for trying to use multiple tools
            bot.history.add_user_event(
                "# System Information\n\n"
                "Error: Found multiple tools. Please select only one tool at a time."
            )

        if "# Thoughts" in response:
            query_counter_message = (
                "# System Message\n\n"
                f"You have {max_num_tool_calls - tool_call_count} tool uses left."
            )
            logger.debug(query_counter_message)
            bot.history.add_user_event(query_counter_message)

        if step_count == max_num_steps - 2:
            bot.history.add_user_event(
                "# System Message\n\nApproaching context limit. Please output READY TO RESPOND now."
            )

        if "READY TO RESPOND" in response:
            break

        if steps_since_last_tool_call >= 3:
            logger.warning("Too many steps since last tool call. Injecting system message.")
            bot.history.add_user_event(
                '# System Message\n\nFriendly reminder to output "READY TO RESPOND" '
                "when you are ready to formulate a response."
            )

        step_count += 1
        steps_since_last_tool_call += 1

    # If we're here, we need to initiate the response sequence.
    bot.history.add_user_event(RESPONSE_INSTRUCTIONS)
    response = bot.complete(max_tokens=1024, temperature=0.2)["content"]
    logger.debug(response)
    final_response = find_markdown_block(response)
    # If the final response is not found, we try again
    if final_response is None:
        logger.warning("Failed to find the final response. Retrying.")
        bot.history.add_user_event(
            "# System Message\n\n"
            "Error: Failed to find the final response. Please try again. Remember to output "
            "your response in a markdown code block under the header '# Final Response'."
        )
        response = bot.complete(max_tokens=1024, temperature=0.2)["content"]
        logger.debug(response)
        final_response = find_markdown_block(response)

    return final_response


def faux_arnar(question: str, *, context: Optional[str], **kwargs):
    bot = Bot(name="faux_arnar", model_name=DEFAULT_FLAGSHIP_LONG_CONTEXT_MODEL)
    user_prompt = bot.format_user_prompt(
        question=question, context=(context or "Not available.")
    )
    bot.history.add_user_event(user_prompt)
    answer = bot.complete(max_tokens=1024, temperature=0.2)["content"]
    return answer.strip()
