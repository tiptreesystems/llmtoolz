import contextlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from common.logger import logger

from althea.arnar import faux_arnar, arnar
from althea.bot import Bot
from althea.lorimer_utils import (
    NOT_FORMULATED_YET,
    NOT_AVAILABLE_THIS_IS_ROOT_SECTION,
    Section,
)

from althea.utils import find_text_under_header, find_markdown_block, DEFAULT_FLAGSHIP_MODEL


def build_executor(parallel: bool, num_threads: int):
    if parallel and num_threads >= 1:
        executor = ThreadPoolExecutor(num_threads)
        map_fn = executor.map
    else:
        executor = contextlib.nullcontext()
        map_fn = map
    return executor, map_fn


def produce_document_tree(
    research_question: str,
    current_depth: int = 0,
    section: Optional[Section] = None,
    parent_section: Optional[Section] = None,
    max_depth: int = 3,
    max_iterations: int = 5,
    parallel: bool = True,
    num_threads: int = 3,
    arnar_kwargs: Optional[dict] = None,
    fake_arnar: bool = False,
) -> Section:
    current_depth += 1

    # Break the recursion if we're in deep enough
    if current_depth > max_depth:
        logger.debug(f"Reached max depth of {max_depth} with section:\n\n{section}")
        return section

    # Create a bot for the forward model
    forward_bot = Bot(
        name="lorimer_forward", model_name=DEFAULT_FLAGSHIP_MODEL, fallback_when_out_of_context=True
    )
    user_prompt = forward_bot.format_user_prompt(
        research_question=research_question,
        section_title=section.title if section else NOT_FORMULATED_YET,
        section_abstract=section.abstract if section else NOT_FORMULATED_YET,
        section_content=section.content if section else NOT_FORMULATED_YET,
        parent_section_title=(
            parent_section.title
            if parent_section
            else NOT_AVAILABLE_THIS_IS_ROOT_SECTION
        ),
        parent_section_abstract=(
            parent_section.abstract
            if parent_section
            else NOT_AVAILABLE_THIS_IS_ROOT_SECTION
        ),
        parent_section_content=(
            parent_section.content
            if parent_section
            else NOT_AVAILABLE_THIS_IS_ROOT_SECTION
        ),
    )
    logger.debug(user_prompt)
    forward_bot.history.add_user_event(user_prompt)

    for step in range(max_iterations):
        # Run the forward model
        response = forward_bot.complete(temperature=0.2, max_tokens=2048)["content"]
        logger.debug(f"Lorimer step {step} response:\n\n{response}")

        # Maybe do research
        if (
            query_string := find_text_under_header(
                response, "# research_assistant.do_research"
            )
        ) is not None:
            query_json = json.loads(query_string)

            if fake_arnar:
                arnar_fn = faux_arnar
            else:
                arnar_fn = arnar

            answer = arnar_fn(
                question=query_json["research_question"],
                context=query_json["research_intent"],
                document_research_question=research_question,
                **arnar_kwargs,
            )

            answer = f"# Response from research_assistant.do_research\n\n```markdown\n{answer}\n```"
            forward_bot.history.add_user_event(answer)
            logger.debug(f"Lorimers research assistant response:\n\n{answer}")

        # If the forward model generated a section,
        elif (section_spec := find_markdown_block(response)) is not None:
            logger.debug(f"Section spec:\n\n{section_spec}")
            if current_depth == max_depth:
                section = Section.from_string(section_spec, parse_subsections=False)
                logger.debug("Maximum recursion depth reached.")
                break
            else:
                section = Section.from_string(section_spec)
                executor, map_fn = build_executor(parallel, num_threads)
                with executor:
                    subsections = list(
                        map_fn(
                            lambda subsection: produce_document_tree(
                                research_question=research_question,
                                current_depth=current_depth,
                                section=subsection,
                                parent_section=section,
                                max_depth=max_depth,
                                max_iterations=max_iterations,
                                parallel=parallel,
                                num_threads=num_threads,
                                arnar_kwargs=dict(arnar_kwargs),
                            ),
                            list(section.subsections),
                        )
                    )
                    section.subsections = subsections
                break

        # Inject a message if we're almost out of steps
        if max_iterations - 2 == step:
            message = "# System Message\n\nYou are *not* permitted to do more research. Emit the final markdown block."
            logger.debug(message)
            forward_bot.history.add_user_event(message)
    return section


def update_abstracts_from_children(research_question: str, section: Section) -> Section:
    # run recursive update on children sections
    for i, subsection in enumerate(section.subsections):
        section.subsections[i] = update_abstracts_from_children(
            research_question, subsection
        )

    logger.debug(f"Updating section: {section.title}")
    # Once all children are updated, then we can update this section
    bot = Bot(name="lorimer_backward", model_name=DEFAULT_FLAGSHIP_MODEL, fallback_when_out_of_context=True)
    user_prompt = bot.format_user_prompt(
        research_question=research_question,
        section_title=section.title,
        section_content=section.content,
        section_sources=section.render_sources(),
    )
    bot.history.add_user_event(user_prompt)
    response = bot.complete(max_tokens=2048, temperature=0.2)["content"]

    if (section_string := find_markdown_block(response)) is not None:
        logger.debug(section_string)
        new_section = Section.from_string(section_string)
        section.title = new_section.title
        section.abstract = new_section.abstract
    logger.debug(response)
    return section


def provide_feedback(query: str, section: Section, num_threads: int = 5) -> Section:
    bot = Bot(name="lorimer_feedback", model_name=DEFAULT_FLAGSHIP_MODEL)
    user_prompt = bot.format_user_prompt(
        query=query,
        all_sections=section.render_as_text(show_abstracts=False),
        current_section=section.title,
    )

    bot.history.add_user_event(user_prompt)
    response = bot.complete()["content"]
    logger.debug(response)

    if response.startswith("# Feedback"):
        # the feedback should have been in a markdown block, so we make it a markdown block
        response = f"```markdown\n{response}\n```"

    if (feedback := find_markdown_block(response)) is not None:
        logger.debug(feedback)
        section.feedback = feedback

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        section.subsections = list(
            executor.map(
                lambda subsection: provide_feedback(query=query, section=subsection),
                section.subsections,
            )
        )

    return section


def update_from_feedback(query: str, section: Section, num_threads: int = 5) -> Section:
    bot = Bot(name="lorimer_update_from_feedback", model_name=DEFAULT_FLAGSHIP_MODEL)

    user_prompt = bot.format_user_prompt(
        query=query,
        section_title=section.title,
        section_abstract=section.abstract,
        section_content=section.content,
        section_sources=section.render_sources(),
        feedback=section.feedback,
    )

    bot.history.add_user_event(user_prompt)
    response = bot.complete()["content"]
    if response.startswith("## Abstract") or response.startswith("## Content"):
        # This is a hack to deal with the fact that the model sometimes does not put it in a markdown block
        response = f"```markdown\n{response}\n```"

    if (section_string := find_markdown_block(response)) is not None:
        logger.debug(section_string)
        new_section = Section.from_string(section_string)

        section.abstract = (
            new_section.abstract if new_section.abstract else section.abstract
        )
        section.content = (
            new_section.content if new_section.content else section.content
        )
        section.sources = (
            new_section.sources if new_section.sources else section.sources
        )

    # Recursively update each section from its feedback
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        section.subsections = list(
            executor.map(
                lambda subsection: update_from_feedback(
                    query=query, section=subsection,
                ),
                section.subsections,
            )
        )

    return section


def lorimer(
    question: str,
    *,
    context: Optional[str] = None,
    max_document_depth: int = 2,
    parallel: bool = True,
    num_threads: int = 3,
    max_iterations: int = 5,
    fake_it: bool = False,
    log_path: Optional[str] = None,
    # Kwargs for arnar
    num_bjork_calls_in_arnar: int = 3,
    arnar_kwargs: Optional[dict] = None,
    # Kwargs for bjork
    parallel_bjork: bool = False,
    num_threads_for_bjork: int = 3,
    max_queries_in_bjork: int = 3,
    num_pages_to_select_in_bjork: int = 3,
    bjork_kwargs: Optional[dict] = None,
) -> Section:
    bjork_kwargs = dict(
        max_queries=max_queries_in_bjork,
        num_pages_to_select=num_pages_to_select_in_bjork,
        parallel=parallel_bjork,
        num_threads=num_threads_for_bjork,
        **(bjork_kwargs or {}),
    )

    arnar_kwargs = dict(
        max_num_bjork_calls=num_bjork_calls_in_arnar,
        bjork_kwargs=bjork_kwargs,
        **(arnar_kwargs or {}),
    )

    question_with_context = (
        f"{question} \n\nThe context for this question is: {context}"
    )
    root_section = produce_document_tree(
        research_question=question_with_context,
        max_depth=max_document_depth,
        parallel=parallel,
        num_threads=num_threads,
        arnar_kwargs=arnar_kwargs,
        max_iterations=max_iterations,
        fake_arnar=fake_it,
    )

    def produce_research_statement(question_with_context: str):
        bot = Bot(name="lorimer_research_statement", model_name=DEFAULT_FLAGSHIP_MODEL)
        user_prompt = bot.format_user_prompt(question=question_with_context)
        bot.history.add_user_event(user_prompt)
        response = bot.complete(max_tokens=1024, temperature=0.2)["content"]
        research_statement = response.split("# Research Statement")[1].strip()
        return research_statement

    research_statement = produce_research_statement(question_with_context)
    root_section.content = "## Research Statement\n" + research_statement + "\n\n" + root_section.content

    if log_path:
        root_section.persist(log_path + "_forward.json")

    root_section = update_abstracts_from_children(
        research_question=question_with_context, section=root_section
    )
    if log_path:
        root_section.persist(log_path + "_backward.json")

    root_section = provide_feedback(query=question_with_context, section=root_section)
    if log_path:
        root_section.persist(log_path + "_feedback.json")
    root_section = update_from_feedback(
        query=question_with_context, section=root_section
    )
    if log_path:
        root_section.persist(log_path + "_updated.json")

    return root_section
