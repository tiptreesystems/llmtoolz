import json
from typing import Optional, List, Union, Tuple

import openai.error
from sklearn.mixture import GaussianMixture
import numpy as np
from common.logger import logger

from bot import Bot
from fuzzysearch import find_near_matches
from functools import reduce

from utils import (
    count_tokens_in_str,
    CONTEXT_LENGTHS,
    DEFAULT_FLAGSHIP_MODEL,
    DEFAULT_VISION_MODEL,
    DEFAULT_FAST_MODEL,
)


FAILED_TO_PARSE_JSON = "Internal error: Failed to parse JSON"
FAILED_TO_FIND_MATCHES = "Internal error: failed to find any keyword matches."
FAILED_TO_RETRIEVE = "Failed to find an answer in provided text."
FAILED_ANSWER_STRING = "Answer was not found. Here is what was found instead: {}"


def extract_keywords(
    question: str,
    model_name: str = DEFAULT_FAST_MODEL,
) -> List[str]:
    bot = Bot(name="question_to_keywords", model_name=model_name)
    user_prompt = bot.format_user_prompt(question=question)
    bot.history.add_user_event(user_prompt)
    # The first step is to clarify the intention of the client and to get a new question
    keywords = bot.complete(max_tokens=128, temperature=0.2)["content"]
    # Split by comma, strip, and return
    keywords = [
        keyword.replace('"', "").strip(" ") for keyword in keywords.strip().split(",")
    ]
    logger.debug(f"Keywords: {keywords}")
    return keywords


def find_matches(
    keywords: list, content: str, max_distance_lambda: float = 0.25
) -> list:
    # delete empty string from keywords
    keywords = [keyword.lower() for keyword in keywords if keyword != ""]
    matches = {}
    for keyword in keywords:
        matches[keyword] = find_near_matches(
            keyword.lower(),
            content.lower(),
            max_l_dist=max(1, round(max_distance_lambda * len(keyword))),
        )
    all_matches = reduce(lambda a, b: a + b, matches.values())
    return sorted(all_matches, key=lambda x: x.start)


def extract_answer(
    question: str, excerpt: str, model_name: str = DEFAULT_FLAGSHIP_MODEL
) -> Tuple[str, bool]:
    def record_answer(answer_is_found: bool, answer: str):
        """
        This function records whether the answer was found or not, and if it was found, what the answer was.

        Args:
            answer_is_found (bool): Whether the answer was found or not.
            answer (str): The answer, if it was found. Can be an empty string if the answer was not found.
        """
        return dict(answer_is_found=answer_is_found, answer=answer)

    bot = Bot(name="excerpt_to_answer", model_name=model_name)
    bot.register_function(record_answer)
    user_prompt = bot.format_user_prompt(question=question, excerpt=excerpt)
    bot.history.add_user_event(user_prompt)
    bot.complete(max_tokens=1024, temperature=0.1)
    answer_dict = bot.call_requested_function(add_to_history=False)
    if answer_dict is None:
        event = bot.history.get_most_recent_assistant_event()
        logger.error(
            f"Failed to parse event {event['content']} "
            f"as function call: {answer_dict}"
        )
        return FAILED_TO_RETRIEVE, False
    return answer_dict["answer"], answer_dict["answer_is_found"]


def in_context_retrieve(
    question: str, content: str, model_name: str
) -> Tuple[str, bool]:
    bot = Bot(
        name="web_page_ask", model_name=model_name, fallback_when_out_of_context=True
    )
    user_prompt = bot.format_user_prompt(question=question, main_content=content)
    logger.debug(user_prompt[:100] + "...")
    bot.history.add_user_event(user_prompt)
    response = bot.complete(max_tokens=1024, temperature=0.1)["content"]
    logger.debug(response)

    # Parse the response as json
    def parse_json(s):
        if s.startswith("```json"):
            s = s.lstrip("```json").rstrip("```")
        elif s.startswith("```"):
            s = s.strip("```")
        return json.loads(s)

    try:
        response = parse_json(response)
    except json.decoder.JSONDecodeError as e:
        # Try once again with the error
        exception_str = str(e)
        bot.history.add_user_event(
            f"Could not parse response as JSON: {exception_str}. \n\n"
            f"Please try again, and remember that your answer should only contain "
            f"valid JSON."
        )
        response = bot.complete(max_tokens=1024, temperature=0.1)["content"]
        try:
            response = json.loads(response)
        except json.decoder.JSONDecodeError as e:
            logger.exception("Failed to parse json from model.")
            # If we still can't parse the response, just return None
            return FAILED_TO_PARSE_JSON, False
    # If we're here, we could parse the response as JSON
    return response["answer"], response["text_is_relevant"]


def vision_retrieve(
    question: str, screenshot_url: str, model_name: str = DEFAULT_VISION_MODEL
) -> Tuple[str, bool]:
    bot = Bot(name="vision_retriever", model_name=model_name)
    user_prompt = bot.format_user_prompt(question=question)
    bot.history.add_user_event(user_prompt, image_url=screenshot_url)

    try:
        response = bot.complete(max_tokens=1024, temperature=0.2)["content"]
    except openai.error.InvalidRequestError as e:
        if "Invalid image" in str(e):
            logger.exception(f"Invalid image url: {screenshot_url}")
            return "Internal error: Invalid image url", False
        else:
            raise

    logger.debug(response)

    def parse_json(s):
        if s.startswith("```json"):
            s = s.lstrip("```json").rstrip("```")
        elif s.startswith("```"):
            s = s.strip("```")
        return json.loads(s)

    # Parse the response as json
    try:
        response = parse_json(response)
    except json.decoder.JSONDecodeError as e:
        # Try once again with the error
        exception_str = str(e)
        bot.history.add_user_event(
            f"Could not parse response as JSON: {exception_str}. \n\n"
            f"Please try again, and remember that your answer should only contain "
            f"valid JSON."
        )
        response = bot.complete(max_tokens=1024, temperature=0.1)["content"]
        try:
            response = parse_json(response)
        except json.decoder.JSONDecodeError as e:
            logger.exception("Failed to parse json from model.")
            # If we still can't parse the response, just return None
            return FAILED_TO_PARSE_JSON, False
    # If we're here, we could parse the response as JSON
    return response["answer"], response["screenshot_is_relevant"]


def fused_vision_in_context_retrieve(
    question: str,
    content: str,
    screenshot_url: Optional[str] = None,
    in_context_retrieve_model_name: str = DEFAULT_FLAGSHIP_MODEL,
    vision_model_name: str = DEFAULT_VISION_MODEL,
    fusion_model_name: str = DEFAULT_FLAGSHIP_MODEL,
) -> Tuple[str, bool]:
    # Retrieve in context
    in_context_answer, in_context_successful = in_context_retrieve(
        question=question, content=content, model_name=in_context_retrieve_model_name
    )
    # Retrieve with vision
    vision_answer, vision_successful = vision_retrieve(
        question=question, screenshot_url=screenshot_url, model_name=vision_model_name
    )
    # We'll need to fuse the answers. But before that, let's add some sauce to the answers
    if not in_context_successful:
        full_in_context_answer = (
            f"Failed to find the answer.\n\n"
            f"Here is what was found instead: {in_context_answer}"
        )
    else:
        full_in_context_answer = in_context_answer
    if not vision_successful:
        full_vision_answer = (
            f"Failed to find the answer.\n\n"
            f"Here is what was found instead: {vision_answer}"
        )
    else:
        full_vision_answer = vision_answer

    bot = Bot(name="vision_in_context_fuser", model_name=fusion_model_name)
    user_prompt = bot.format_user_prompt(
        question=question,
        in_context_answer=full_in_context_answer,
        vision_answer=full_vision_answer,
    )
    logger.debug(user_prompt)
    bot.history.add_user_event(user_prompt)
    response = bot.complete(max_tokens=1024, temperature=0.2)["content"]
    logger.debug(response)
    # The answer is found if either the in-context or the vision model found the answer
    answer_is_found = in_context_successful or vision_successful
    return response, answer_is_found


def find_retrieve(
    question: str,
    content: str,
    answer_model_name: str = DEFAULT_FLAGSHIP_MODEL,
    keyword_model_name: str = DEFAULT_FLAGSHIP_MODEL,
    max_excerpt_num_chars: int = 16000,
    num_modes_to_try: int = 5,
) -> Tuple[str, bool]:
    # Possible improvements:
    #   - Consider multiple modes of the GMM
    #   - Use a function call to determine if an answer is found.
    # ---------
    keywords = extract_keywords(question, model_name=keyword_model_name)
    all_matches = find_matches(keywords, content)

    if not all_matches:
        return FAILED_TO_FIND_MATCHES, False

    midpoints = [m.start + (m.end - m.start) // 2 for m in all_matches]
    logger.debug(f"Found {len(midpoints)} matches in the content.")

    # If we only have one match, then we can just return 2k chars around it
    if len(midpoints) < 2:
        excerpt = content[
            max(0, midpoints[0] - 1000) : min(len(content), midpoints[0] + 1000)
        ]
        return extract_answer(question, excerpt, model_name=answer_model_name)

    # Figure out how many components we'll need
    if len(midpoints) < 5:
        # If we don't have enough samples, just use one component
        num_components = 1
    else:
        num_components = 5

    gmm = GaussianMixture(n_components=num_components).fit(
        np.array(midpoints).reshape(-1, 1)
    )
    weights = gmm.weights_.ravel()
    sorted_mode_indices = np.argsort(weights)[::-1]

    processed_ranges = []
    answer = FAILED_TO_RETRIEVE
    success = False
    for idx in range(min(num_modes_to_try, len(sorted_mode_indices))):
        mode_idx = sorted_mode_indices[idx]
        logger.debug(f"Processing mode {mode_idx} with weight {weights[mode_idx]}.")

        mean = gmm.means_.ravel()[mode_idx]
        std = np.sqrt(gmm.covariances_.ravel()[mode_idx])

        excerpt_start = max(int(mean - 2 * std), 0)
        excerpt_end = min(int(mean + 2 * std), len(content))

        # Limit excerpt to 16,000 characters
        excerpt_length = excerpt_end - excerpt_start
        if excerpt_length > max_excerpt_num_chars:
            adjustment = (excerpt_length - max_excerpt_num_chars) // 2
            excerpt_start += adjustment
            excerpt_end -= adjustment

        if (excerpt_start, excerpt_end) in processed_ranges:
            logger.debug(f"Already processed the content, skipping attempt {idx}.")
            continue
        else:
            processed_ranges.append((excerpt_start, excerpt_end))

        excerpt = content[excerpt_start:excerpt_end]

        answer, success = extract_answer(
            question, excerpt, model_name=answer_model_name
        )
        if success:
            logger.debug(f"Answer found at attempt {idx}.")
            break
        logger.debug(f"Answer not found at attempt {idx}. Non-answer: {answer}")

    return answer, success


def construct_return_value(
    answer: str, success: bool, full_return: bool, force_string_return: bool
) -> Union[str, Optional[str], Tuple[str, bool]]:
    if full_return:
        if force_string_return:
            answer = answer if success else FAILED_ANSWER_STRING.format(answer)
        return answer, success
    if force_string_return and not success:
        return FAILED_ANSWER_STRING.format(answer)
    return answer if success else None


def retrieve(
    question: str,
    content: str,
    screenshot_url: Optional[str] = None,
    answer_model_name: str = DEFAULT_FLAGSHIP_MODEL,
    keyword_model_name: str = DEFAULT_FLAGSHIP_MODEL,
    max_excerpt_num_chars: int = 16000,
    num_modes_to_try: int = 5,
    in_context_num_token_threshold: Union[str, int] = "auto",
    max_num_in_context_tokens: int = 6000,
    in_context_retrieve_model_name: str = DEFAULT_FLAGSHIP_MODEL,
    vision_model_name: str = DEFAULT_VISION_MODEL,
    full_return: bool = False,
    force_string_return: bool = False,
) -> Union[Optional[str], Tuple[str, bool]]:
    num_tokens = count_tokens_in_str(content, model_name=in_context_retrieve_model_name)
    if in_context_num_token_threshold == "auto":
        in_context_num_token_threshold = (
            CONTEXT_LENGTHS.get(in_context_retrieve_model_name, 4000) * 0.7
        )
        in_context_num_token_threshold = min(
            in_context_num_token_threshold, max_num_in_context_tokens
        )
    # If the number of tokens is small enough, we can use the in-context model
    if num_tokens <= in_context_num_token_threshold:
        if screenshot_url is not None:
            logger.debug("Using in-context + vision retrieval.")
            answer, success = fused_vision_in_context_retrieve(
                question=question,
                content=content,
                screenshot_url=screenshot_url,
                in_context_retrieve_model_name=in_context_retrieve_model_name,
                vision_model_name=vision_model_name,
                fusion_model_name=in_context_retrieve_model_name,
            )
            return construct_return_value(
                answer, success, full_return, force_string_return
            )
        else:
            logger.debug("Using in-context retrieval.")
            answer, success = in_context_retrieve(
                question=question,
                content=content,
                model_name=in_context_retrieve_model_name,
            )
            return construct_return_value(
                answer, success, full_return, force_string_return
            )
    else:
        answer, success = find_retrieve(
            question=question,
            content=content,
            answer_model_name=answer_model_name,
            keyword_model_name=keyword_model_name,
            max_excerpt_num_chars=max_excerpt_num_chars,
            num_modes_to_try=num_modes_to_try,
        )
        return construct_return_value(answer, success, full_return, force_string_return)
