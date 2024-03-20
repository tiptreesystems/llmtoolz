from typing import Any

from common.logger import logger

from bjork import bjork
from multitool import ToolDefinition, ToolDataType
from utils import DEFAULT_FLAGSHIP_MODEL


RA_TOOL_DOC = """Use this tool if you would like to research a topic. You can ask this tool a question and it will gather information for you."""

RA_ARGUMENT_DOC = """The argument `research_intent` *must* be formulated with three sentences:
(1) what kind of facts you want to find (e.g. quotes, statistics, lists, etc.),
(2) which sources you consider reputable (e.g. a specific website, a specific database, a patent, a corporate filing, etc.),
(3) what you want to do with the information.
The `research_question` precisely specifies what the question is that should be researched. It is VITALLY IMPORTANT that your question is formulated in a way that does not have any ambiguities. Do NOT use acronyms, or shorthand or any other form of language that could be interpreted in multiple ways. Do NOT omit any details that are relevant to the question. Do NOT be vague."""

RA_ARGUMENTS = {
    "research_intent": ToolDataType.STRING,
    "research_question": ToolDataType.STRING,
}


class ResearchAssistant(ToolDefinition):
    def __init__(
        self,
        model_name: str = DEFAULT_FLAGSHIP_MODEL,
        bjork_kwargs: dict = None,
        considered_web_pages_: list = None,
        considered_proprietary_data_: list = None,
    ):
        super().__init__(
            tool_name="research_assistant",
            function_name="do_research",
            tool_doc=RA_TOOL_DOC,
            argument_doc=RA_ARGUMENT_DOC,
            arguments=RA_ARGUMENTS,
        )
        self.model_name = model_name
        self.bjork_kwargs = bjork_kwargs
        self.considered_web_pages_ = considered_web_pages_
        self.considered_proprietary_data_ = considered_proprietary_data_

    def tool_fn(self, arguments: dict) -> Any:
        bjork_context = f"{arguments['research_intent']}"

        bjork_extra_kwargs = dict(
            num_pages_to_select=3,
            max_queries=2,
            validate_query=False,
            pull_and_ask=True,
            reason_about_documents_to_select=True,
            main_loop_model=self.model_name,
            link_selection_model=self.model_name,
            link_synthesis_model=self.model_name,
            summarizer_model=self.model_name,
            tool_selection_model=self.model_name,
        )

        full_bjork_kwargs = {
            **bjork_extra_kwargs,
            **(self.bjork_kwargs or {}),
        }

        try:
            answer = bjork(
                query=arguments["research_question"],
                context=bjork_context,
                considered_web_pages_=self.considered_web_pages_,
                considered_proprietary_data_=self.considered_proprietary_data_,
                **full_bjork_kwargs,
            )
        except Exception as e:
            answer = f"Bjork Exception: {e}"
            logger.exception(answer)

        return answer
