import json
from typing import Optional, List, Tuple

import pandas as pd
from common.logger import logger

from bjork_utils import (
    SearchAndSummarize,
    QueryWebPage,
    PeopleLookup,
)
from bot import Bot
from multitool import (
    ToolBelt,
    find_tool,
)
from scrape import WebPage
from utils import (
    find_text_under_header,
    find_json_block,
    DEFAULT_FLAGSHIP_MODEL,
)


def bjork(
    query: str,
    *,
    context: Optional[str] = None,
    system_prompt: Optional[str] = None,
    validate_query: bool = False,
    max_queries: int = 2,
    num_pages_to_select: int = 3,
    max_num_steps: int = 20,
    main_loop_model: str = DEFAULT_FLAGSHIP_MODEL,
    tool_selection_model: str = DEFAULT_FLAGSHIP_MODEL,
    link_selection_model: str = DEFAULT_FLAGSHIP_MODEL,
    link_synthesis_model: str = DEFAULT_FLAGSHIP_MODEL,
    summarizer_model: str = DEFAULT_FLAGSHIP_MODEL,
    model_name: Optional[str] = None,
    parallel: bool = True,
    num_threads: int = 3,
    pull_and_ask: bool = True,
    reason_about_documents_to_select: bool = True,
    # These lists will be manipulated in place, if provided
    considered_web_pages_: Optional[List["WebPage"]] = None,
) -> str:
    # Validate if needed
    if validate_query:
        reason, can_answer = validate_bjork_query(
            query=query, context=context, default=True
        )
    else:
        reason, can_answer = "N/A", True

    if not can_answer:
        return f"Bad query. Reason: {reason}"

    if model_name is not None:
        main_loop_model = model_name
        tool_selection_model = model_name
        link_selection_model = model_name
        link_synthesis_model = model_name
        summarizer_model = model_name

    # Build the tools
    search_tool = SearchAndSummarize(
        num_pages_to_select=num_pages_to_select,
        link_selection_model=link_selection_model,
        link_synthesis_model=link_synthesis_model,
        summarizer_model=summarizer_model,
        parallel=parallel,
        num_threads=num_threads,
        pull_and_ask=pull_and_ask,
        reason_about_documents_to_select=reason_about_documents_to_select,
        considered_web_pages_=considered_web_pages_,
    )
    web_page_query_tool = QueryWebPage(summarizer_model=summarizer_model)
    people_lookup_tool = PeopleLookup(model_name=main_loop_model)
    # Build the belt
    tool_belt = ToolBelt(
        tools=[
            search_tool,
            web_page_query_tool,
            people_lookup_tool,
        ]
    )

    bot = Bot(
        name="bjork",
        model_name=main_loop_model,
        system_prompt=system_prompt,
        fallback_when_out_of_context=True,
    )

    logger.user("Initiating...")

    if context is not None:
        user_message = f"# Input Query\n\n{query}\n\n# Search Intent\n\n{context}\n\n# Max Queries\n\n{max_queries}\n\n# Current Date\n\n{pd.to_datetime('today').strftime('%B %d, %Y')}"
        logger.user(f'Search query: "{query}"')
        logger.user(f'Search intent: "{context}"')
    else:
        user_message = f"# Input Query\n\n{query}\n\n# Max Queries\n\n{max_queries}\n\n# Current Date\n\n{pd.to_datetime('today').strftime('%B %d, %Y')}"
        logger.user(f'Search query: "{query}"')

    logger.debug(user_message)
    bot.history.add_user_event(user_message)
    query_count = 0
    step_count = 0
    final_response = ""

    while True:
        response = bot.complete(temperature=0.1, max_tokens=1024)["content"]
        logger.debug(response)

        if "# Tool Query" in response:
            if query_count >= max_queries:
                bot.history.add_user_event(
                    "# System Information\n\nError: cannot use the multitool anymore. Please produce final response now."
                )
            else:
                tool_search_query = find_text_under_header(response, "# Tool Query")
                logger.user("Searching for tool...")
                found_tools = find_tool(
                    task=tool_search_query,
                    tools=tool_belt,
                    model_name=tool_selection_model,
                )
                if len(found_tools) > 0:
                    logger.user(f"Found tool: {found_tools[0].tool_name}")
                    lines = ["# Found Tools", ""]
                    for tool in found_tools:
                        lines.append(tool.render())
                        lines.append("")
                    tool_lines = "\n".join(lines)
                    bot.history.add_user_event(tool_lines)

        # Check if a tool is being used
        matching_tools = tool_belt.find_matching_tools(response)

        if len(matching_tools) == 1:
            # Found only one tool, as expected.
            tool = matching_tools[0]
            valid, message = tool.check_call_arguments(response)
            if not valid:
                error_message = (
                    f"Error calling tool {tool.tool_name}.{tool.function_name} "
                    f"-- {message}"
                )
                bot.history.add_user_event(error_message)
            else:
                logger.user(f"Using tool: {tool.tool_name}")
                # Tool is invoked and it's valid, so we use it.
                answer, was_successful = tool.use(response)
                if was_successful:
                    query_count += 1
                    formatted_answer = tool.format_result(answer)
                else:
                    formatted_answer = f"This tool failed to return a result. More information: {answer}"
                    formatted_answer = tool.format_result(formatted_answer)

                logger.user(f"Response from tool: {formatted_answer}")

                formatted_answer = (
                    f"{formatted_answer}\n\n"
                    f"# System Information\n\n"
                    f"Please provide an analysis of the result under the "
                    f"markdown header '# Analysis'."
                )
                # Log and add to history
                logger.debug(formatted_answer)
                bot.history.add_user_event(formatted_answer)

        elif len(matching_tools) > 1:
            # Scold the model for trying to use multiple tools
            bot.history.add_user_event(
                "# System Information\n\n"
                "Error: Found multiple tools. Please select only one tool at a time."
            )

        if "# Analysis" in response:
            query_counter_message = (
                "# System Message\n\n"
                f"You have {max_queries - query_count} queries available."
            )
            logger.debug(query_counter_message)
            bot.history.add_user_event(query_counter_message)

            # Get the analysis text for logging
            analysis_text = find_text_under_header(response, "# Analysis")
            logger.user(f"Analyzing...")
            logger.user(f"{analysis_text}")

        if step_count == max_num_steps - 2:
            bot.history.add_user_event(
                "# System Message\n\nApproaching context limit. Please produce final response now."
            )

        if "# Final Response" in response:
            final_response = response.split("# Final Response")[1]
            break

        step_count += 1

        if step_count == max_num_steps:
            break

    logger.user("Preparing final response...")
    logger.user(final_response)
    return final_response


def validate_bjork_query(
    query: str,
    context: Optional[str] = None,
    default: bool = True,
    model_name: str = DEFAULT_FLAGSHIP_MODEL,
) -> Tuple[str, bool]:
    bot = Bot(name="bjork_validator", model_name=model_name)
    if context is not None:
        context = "Not available."
    user_message = bot.format_user_prompt(
        question=query,
        context=context,
        current_date=pd.to_datetime("today").strftime("%B %d, %Y"),
    )
    bot.history.add_user_event(user_message)
    response = bot.complete(max_tokens=1024, temperature=0.1)["content"]
    logger.debug(response)
    validation_result = None
    # Find the json and validate it
    try:  # Try to parse the json
        validation_result = find_json_block(response, load=True)
    except json.decoder.JSONDecodeError:
        # Oop
        pass
    if validation_result is None:
        return "Failed to validate query.", default
    else:
        return validation_result["reason"], validation_result["can_answer"]
