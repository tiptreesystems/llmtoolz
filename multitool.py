import json
from dataclasses import dataclass
from enum import Enum
from json import JSONDecodeError
from typing import Dict, List, Optional, Any, Tuple

from common.logger import logger

from bot import Bot
from utils import find_text_under_header, find_json_block, DEFAULT_FLAGSHIP_MODEL


class ToolDataType(Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    NULL = "null"
    UNDEFINED = "undefined"
    ANY = "any"


class ToolDefinition:
    def __init__(
        self,
        tool_name: str,
        tool_doc: str,
        function_name: str,
        argument_doc: str,
        arguments: Dict[str, ToolDataType],
        post_use_message: Optional[str] = None,
    ):
        self.tool_name = tool_name
        self.tool_doc = tool_doc
        self.function_name = function_name
        self.argument_doc = argument_doc
        self.arguments = arguments
        self.post_use_message = post_use_message

    def tool_fn(self, arguments: dict) -> Any:
        raise NotImplementedError

    def check_if_call_requested(self, text: str) -> bool:
        return f"# {self.tool_name}.{self.function_name}" in text

    def check_call_arguments(self, text: str) -> Tuple[bool, str]:
        if not self.check_if_call_requested(text):
            return (
                False,
                f"Call is not requested to {self.tool_name}.{self.function_name}",
            )
        # Parse call arguments
        arguments = self.parse_call_arguments(text)
        expected_types = {
            ToolDataType.STRING: str,
            ToolDataType.NUMBER: (int, float),
            ToolDataType.BOOLEAN: bool,
        }
        for key, val in arguments.items():
            expected_type = expected_types.get(self.arguments[key])
            if expected_type is None:
                continue
            if not isinstance(val, expected_type):
                message = (
                    f"Error parsing argument for function {self.tool_name}.{self.function_name}. "
                    f"Argument {key} is not of type {self.arguments[key].value}, "
                    f"but is of type {type(val)}."
                )
                return False, message
        return True, "Call arguments are valid."

    def parse_call_arguments(self, text: str) -> Optional[dict]:
        if not self.check_if_call_requested(text):
            return None
        # Remove all backticks
        text = text.strip("```")
        # Get the arguments
        arguments = find_text_under_header(
            text, f"# {self.tool_name}.{self.function_name}"
        )
        if arguments is None:
            return None
        arguments = arguments.strip("```")
        arguments = json.loads(arguments)
        return arguments

    def use(self, text: str) -> Optional[Any]:
        if not self.check_if_call_requested(text):
            return None
        # Get the arguments
        arguments = self.parse_call_arguments(text)
        if arguments is None:
            return None
        # Call the tool function
        output = self.tool_fn(arguments)
        return output

    def render(self):
        lines = [
            f"## {self.tool_name}",
            f"",
        ]
        # Render the tool doc lines
        tool_doc_lines = [f"// {line}" for line in self.tool_doc.split("\n")]
        lines.extend(tool_doc_lines)
        # Add namespace line
        lines.append(f"namespace {self.tool_name} {{")
        # Blank line
        lines.append("")
        # Render the function doc lines
        argument_doc_lines = [f"// {line}" for line in self.argument_doc.split("\n")]
        lines.extend(argument_doc_lines)
        # Add function line
        lines.append(f"type {self.function_name} = (_: {{")
        # Render the tool lines
        for argument, data_type in self.arguments.items():
            lines.append(f"  {argument}: {data_type.value},")
        # Close the function
        lines.append("}) => any;")
        # Close the namespace
        lines.append(f"}} // namespace {self.tool_name}")
        # Return the lines
        return "\n".join(lines)

    def format_result(self, result: str):
        lines = [f"# Result from {self.tool_name}.{self.function_name}", "", result]
        return "\n".join(lines)


@dataclass
class ToolBelt:
    tools: List[ToolDefinition]

    def render(self):
        tool_renders = [tool.render() for tool in self.tools]
        return "\n\n".join(tool_renders)

    def find_matching_tools(self, text: str) -> List[ToolDefinition]:
        matching_tools = []
        for tool in self.tools:
            if tool.check_if_call_requested(text):
                matching_tools.append(tool)
        return matching_tools

    def __len__(self):
        return len(self.tools)

    def __getitem__(self, idx):
        return self.tools[idx]

    def __iter__(self):
        return iter(self.tools)

    def __contains__(self, item):
        return item in self.tools


@dataclass
class ToolUse:
    tool: ToolDefinition
    arguments: Dict[str, Any]
    result: Any

    def format_answer(self, answer: Optional[str] = None):
        if answer is None:
            answer = f"```json\n{json.dumps(self.result, indent=2)}\n```"

        lines = [
            "# Tool Response",
            "",
            f"## Selected Tool",
            "",
            f"**Tool Name:** {self.tool.tool_name}.{self.tool.function_name}",
            f"",
            f"**Tool Documentation:** {self.tool.tool_doc}",
            f"",
            f"**Argument Documentation:** {self.tool.argument_doc}",
            f"",
            f"Arguments used:",
            f"```json\n{json.dumps(self.arguments, indent=2)}\n```",
            f"",
            "## Tool Output",
            f"",
            answer,
            f"",
        ]
        return "\n".join(lines)


def find_tool(
    task: str,
    tools: ToolBelt,
    num_iterations: int = 3,
    model_name: str = DEFAULT_FLAGSHIP_MODEL,
) -> List[ToolDefinition]:
    # If there's only a single tool in the belt, we return it right away to
    # save tokens.
    if len(tools) == 1:
        return [tools[0]]
    bot = Bot(name="tool_finder", model_name=model_name)
    user_prompt = bot.format_user_prompt(task=task, tools=tools.render())
    bot.history.add_user_event(user_prompt)
    logger.debug(user_prompt)
    matching_tools = []
    for iter_num in range(num_iterations):
        response = bot.complete(max_tokens=1024, temperature=0.2)["content"]
        logger.debug(response)
        # Find the matching tools
        try:
            tool_identifier = find_json_block(response, load=True)
        except JSONDecodeError as e:
            logger.warning("Failed to use the right syntax. Retrying.")
            tool_identifier = None
            bot.history.add_user_event(
                "# System Message\n\n"
                f"The syntax was not correct. Parsing error: {str(e)}.\n\n"
                f"Please try again"
            )
        if tool_identifier is not None:
            try:
                tool_name, function_name = tool_identifier["tool"].split(".")
            except ValueError as e:
                logger.error(f"Failed to parse tool identifier. Error: {str(e)}")
                bot.history.add_user_event(
                    "# System Message\n\n"
                    "The tool identifier is not in the correct format. "
                    "Please use the format `tool_name.function_name`. Remember, you must pick a tool."
                )
                continue
            matching_tools = tools.find_matching_tools(f"# {tool_name}.{function_name}")
        # Break if we found any matching tools
        if len(matching_tools) > 0:
            break
        # Give it a chance to find the tools
        if iter_num + 2 == num_iterations:
            logger.warning("Reprompting for tool search.")
            bot.history.add_user_event(
                "# System Message\n\nApproaching iteration limit. "
                "Please produce the result now as a json block under "
                'the header "# Tool Selection".'
            )
    return matching_tools


def multitool(task: str, tools: ToolBelt, num_iterations: int = 5) -> Optional[ToolUse]:
    bot = Bot(name="multitool", model_name=DEFAULT_FLAGSHIP_MODEL)
    user_prompt = bot.format_user_prompt(task=task, tools=tools.render())
    bot.history.add_user_event(user_prompt)
    logger.debug(user_prompt)
    for iter_num in range(num_iterations):
        response = bot.complete(max_tokens=1024, temperature=0.2)["content"]
        logger.debug(response)
        # Find the matching tools
        matching_tools = tools.find_matching_tools(response)
        if len(matching_tools) == 0:
            # Reprompt if needed
            if iter_num == num_iterations - 2:
                logger.warning("Reprompting for tool use.")
                bot.history.add_user_event(
                    "# System Message\n\nApproaching iteration limit. Please use a tool now."
                )
            continue
        tool = matching_tools[0]
        # Check if tool valid
        valid, message = tool.check_call_arguments(response)
        if not valid:
            error_message = (
                f"Error calling tool {tool.tool_name}.{tool.function_name} -- {message}"
            )
            logger.error(error_message)
            # Tell this to the LLM and maybe try again
            bot.history.add_user_event(f"# System Message\n\n{error_message}")
            continue
        result = tool.use(response)
        return ToolUse(
            tool=tool, result=result, arguments=tool.parse_call_arguments(response)
        )
