import base64
import glob
import io
import pickle
import shutil
import signal
import string
import threading
import time
import uuid
import mistune
import pdfkit
from contextlib import contextmanager
from dataclasses import dataclass, field
import ruptures as rpt

import re
import tiktoken
import openai
import numpy as np
from PIL import Image
from packaging import version
from urllib.parse import urlparse, urlunparse
from diskcache import Cache
from functools import wraps
from typing import List, Dict, Tuple, IO, Union

from common.utils import *

# ------ Python-fu ------


class Singleton(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                # If the instance doesn't exist, create it and store it in the _instances dictionary.
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        # Return the singleton instance.
        return cls._instances[cls]


def format_dict(dictionary, indent=0):
    result = ""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            result += "  " * indent + f"{key}:\n"
            result += format_dict(value, indent + 1)
        else:
            result += "  " * indent + f"{key}: {value}\n"
    return result


def flatten_dict(d, parent_key="", sep="/"):
    """
    Flatten an arbitrary nested dictionary.

    Parameters:
    - d (dict): The dictionary to flatten.
    - parent_key (str, optional): The base key for recursive calls. Defaults to ''.
    - sep (str, optional): The separator to use between keys. Defaults to '/'.

    Returns:
    - dict: The flattened dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def format_string(s: str, **kwargs):
    # Parse the format string to get the field names
    formatter = string.Formatter()
    field_names = [
        field_name
        for _, field_name, _, _ in formatter.parse(s)
        if field_name is not None
    ]

    # Create a dictionary with only the required fields
    required_kwargs = {key: kwargs[key] for key in field_names if key in kwargs}

    return s.format(**required_kwargs)


def multiline_input(prompt: Optional[str] = "auto"):
    contents = []
    if prompt is None:
        pass
    elif prompt == "auto":
        print("Enter your text below. Type :wq to finish.")
    else:
        print(prompt)
    while True:
        line = input()
        if line == ":wq":
            break
        contents.append(line)
    return "\n".join(contents)


@dataclass
class TimestampedString:
    timestamp: float
    string: str

    @classmethod
    def from_string(cls, string: str):
        return cls(timestamp=time.time(), string=string)

    def get_age(self, unit: str = "hours"):
        if unit == "hours":
            return (time.time() - self.timestamp) / 3600
        elif unit == "days":
            return (time.time() - self.timestamp) / (3600 * 24)
        elif unit == "weeks":
            return (time.time() - self.timestamp) / (3600 * 24 * 7)
        elif unit == "months":
            return (time.time() - self.timestamp) / (3600 * 24 * 30)
        elif unit == "years":
            return (time.time() - self.timestamp) / (3600 * 24 * 365)
        else:
            raise ValueError(f"Invalid unit: {unit}")

    def __str__(self):
        return self.string

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "string": self.string,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            timestamp=d["timestamp"],
            string=d["string"],
        )

    def __hash__(self):
        return hash((self.timestamp, self.string))


def markdown_to_html(text):
    import re

    # Convert markdown text to HTML with updated handling for footer links and quotes
    lines = text.split("\n")
    html = []

    # Regular expressions for different markdown features
    bold_regex = r"\*\*(.*?)\*\*"
    italic_regex = r"\*(.*?)\*"
    bullet_point_regex = r"^[\*-] (.*)"
    enumeration_regex = r"^\d+\.\s(.*)"
    link_regex = r"\[([^\]]+)\]\(([^\)]+)\)"
    footer_link_regex = r"\[\^([^\]]+)\^\]:\s?\[(.*?)\]\((.*?)\)"
    footnote_ref_regex = r"\[\^(\d+)\^\]"
    quote_regex = r"^> (.*)"
    header_3_regex = r"^###\s?(.*)"  # Regex for "### Header 3"
    header_2_regex = r"^##\s?(.*)"  # Regex for "## Header 2"
    header_1_regex = r"^#\s?(.*)"  # Regex for "# Header 1"

    for line in lines:
        if line:
            # Replace footnote references like [^1^] with [1]
            line = re.sub(footnote_ref_regex, r"[\1]", line)

            # Replace markdown syntax with HTML tags
            line = re.sub(bold_regex, r"<b>\1</b>", line)
            line = re.sub(italic_regex, r"<i>\1</i>", line)
            line = re.sub(bullet_point_regex, r"<li>\1</li>", line)
            line = re.sub(enumeration_regex, r"<li>\1</li>", line)
            line = re.sub(link_regex, r'<a href="\2">\1</a>', line)
            line = re.sub(footer_link_regex, r'<a name="\1" href="\3">\2</a>', line)
            line = re.sub(header_3_regex, r"<h3>\1</h3>", line)  # Handle "### Header 3"
            line = re.sub(header_2_regex, r"<h2>\1</h2>", line)  # Handle "## Header 2"
            line = re.sub(header_1_regex, r"<h1>\1</h1>", line)  # Handle "# Header 1"

            # Handle quotes
            if re.match(quote_regex, line):
                line = re.sub(
                    quote_regex,
                    r'<div style="border-left: 2px solid #ccc; color: #666; margin-left: 10px; padding-left: 10px;">\1</div>',
                    line,
                )

            # Wrap in paragraph tag if not a list item, quote, or header
            if not (
                line.startswith("<li>")
                or line.startswith("<div style")
                or line.startswith("<h2>")
                or line.startswith("<h3>")
                or line.startswith("<h1>")
            ):
                line = f"<p>{line}</p>"

            html.append(line)

    body = "".join(html)
    full_html = f"<html><head></head><body>{body}</body></html>"
    return full_html


def truncate_after_markdown_header_with_flag(
    text: str,
    header: str,
    truncation_str: str = "[...]",
    truncate_subheaders: bool = True,
) -> str:
    """
    This function truncates the text after the first section under the specified markdown header.
    It includes a flag to determine if subheaders should also be truncated.
    """

    # Split the text by lines to process headers
    lines = text.split("\n")
    # Variable to store the index of line where the header is found
    start_index = None
    # Variable to store the index of line where the next header of the same or higher level is found
    end_index = None

    # Search for the line index of the specified header
    for i, line in enumerate(lines):
        if line.strip() == header:
            start_index = i
            break

    # If the header was not found, return the original text
    if start_index is None:
        return text

    # Determine the level of the original header
    original_header_level = header.count("#")

    # Search for the next header of the same or higher level
    for i, line in enumerate(lines[start_index + 1 :], start=start_index + 1):
        # Check if the line starts with a header
        if line.startswith("#"):
            current_header_level = line.count("#")
            # If subheaders should not be truncated and the current header level is greater than the original, continue
            if not truncate_subheaders and current_header_level > original_header_level:
                continue
            # If the level is the same or higher than the original header, mark the end index
            if current_header_level <= original_header_level:
                end_index = i
                break

    # If the end index wasn't found, we truncate everything after the start header
    end_index = end_index or len(lines)

    # Join the text back together with the truncation after the specified header's section
    truncated_text = "\n".join(
        lines[: start_index + 1] + [truncation_str] + lines[end_index:]
    )

    return truncated_text


def find_markdown_header(text: str, header: str) -> bool:
    # Make sure header starts with a '#'
    assert header.startswith("#"), "Header must start with '#'"
    # Escape special characters in the header text
    header_text = re.escape(header.lstrip("#").strip())

    # Create a regex pattern that matches the header with the correct number of '#' characters
    pattern = rf'^\s*{"#"*header.count("#")} {header_text}\s*$'

    # Search for the pattern in the text using MULTILINE flag
    return bool(re.search(pattern, text, re.MULTILINE))


def find_text_under_header(
    text: str, header: str, keep_subheaders: bool = False, assert_found: bool = False
) -> Optional[str]:
    if not header.startswith("#"):
        return None

    if not find_markdown_header(text, header):
        return None

    lines = text.split("\n")
    header_level = header.count("#")
    found_header = False
    result = []

    for line in lines:
        if line.startswith(header):
            found_header = True
            continue

        if found_header:
            if keep_subheaders:
                break_strings = ["#" * i + " " for i in range(1, header_level + 1)]
            else:
                break_strings = ["#"]

            if any(line.startswith(break_string) for break_string in break_strings):
                break

            result.append(line)

    if not result:
        if assert_found:
            raise ValueError(f"Header '{header}' not found in text:\n\n{text}")
        else:
            return None

    return "\n".join(result)


def find_markdown_block(text: str) -> Optional[str]:
    # Pattern to match markdown code block
    pattern = r"```markdown(.*?)```"

    # Searching for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    # Returning the matched group if found, else an empty string
    return match.group(1).strip() if match else None


def file_to_base64(file_path_or_file: Union[str, IO]) -> str:
    if isinstance(file_path_or_file, str):
        file_path = file_path_or_file
        file = open(file_path, "rb")
    else:
        file = file_path_or_file
    base64_str = base64.b64encode(file.read()).decode("utf-8")
    if isinstance(file_path_or_file, str):
        file.close()
    return base64_str


def image_to_base64(image_path_or_image_file: Union[str, IO]) -> str:
    base64_image = file_to_base64(image_path_or_image_file)
    return f"data:image/jpeg;base64,{base64_image}"


def read_base64_url_image(base64_url_str: str) -> np.ndarray:
    # Split the base64 URL to extract the base64 string part
    base64_str = base64_url_str.split(",")[1]

    # Decode the base64 string to get raw image bytes
    image_bytes = base64.b64decode(base64_str)

    # Read the image from bytes using PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Convert the PIL image to a NumPy array
    numpy_image = np.array(image)

    return numpy_image


def find_json_block(text: str, load: bool = False) -> Optional[Union[str, dict, list]]:
    # Pattern to match JSON code block
    pattern = r"```json(.*?)```"

    # Searching for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    # Returning the matched group if found, else an empty string
    json_block = match.group(1).strip() if match else None

    if load:
        return json.loads(json_block) if json_block is not None else None
    else:
        return json_block


def parse_json(text: str, raise_on_fail: bool = True) -> Optional[Union[dict, list]]:
    if "```json" in text and "```" in text:
        return find_json_block(text, load=True)
    if text.startswith("```"):
        text.strip("```")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if raise_on_fail:
            raise
        else:
            return None


class TimeoutException(Exception):
    pass


def timeout(seconds):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(
                f"Function '{func.__name__}' timed out after {seconds} seconds"
            )

        def wrapper(*args, **kwargs):
            # Set the signal handler and a timeout
            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the original signal handler
                signal.signal(signal.SIGALRM, old_handler)
                # Cancel the alarm
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def parse_prompts(input_data):
    # Check if the input is a file path
    if os.path.exists(input_data):
        with open(input_data, "r") as file:
            markdown_text = file.read()
    else:
        markdown_text = input_data

    # Use a regular expression to split on ':::user' only when it is on a line by itself
    parts = re.split(r"\n\s*:::user\s*\n", markdown_text, maxsplit=1)

    # Strip leading and trailing whitespace from each part
    system_prompt = parts[0].strip()
    user_prompt = parts[1].strip() if len(parts) > 1 else None

    return system_prompt, user_prompt


def find_prompt_path(name: str, version_string: Optional[str] = None):
    prompt_root = get_path("althea/bots")

    # First case: name is a path
    candidate_path = os.path.join(prompt_root, f"{name}.md")
    if os.path.exists(candidate_path):
        return candidate_path

    # Second case: name is a directory
    candidate_path = os.path.join(prompt_root, name)
    if version_string is None and os.path.isdir(candidate_path):
        # List all Markdown files in the directory
        files = [f for f in os.listdir(candidate_path) if f.endswith(".md")]

        # Sort files based on version numbers
        sorted_files = sorted(
            files, key=lambda f: version.parse(f.split("v")[-1].replace(".md", ""))
        )

        # Return the path of the most recent version
        if sorted_files:
            return os.path.join(candidate_path, sorted_files[-1])

    # Third case: name is a directory with a version preference
    elif version_string is not None:
        assert os.path.isdir(candidate_path)
        # Find the file
        prompt_path = os.path.join(candidate_path, f"v{version_string}.md")
        if os.path.exists(prompt_path):
            return prompt_path
        else:
            raise ValueError(f"Version {version_string} not found in {candidate_path}")

    else:
        raise ValueError(
            f"Prompt {name} (version {version_string}) not found in {prompt_root}"
        )


SESSION_ID = f"s{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:5]}"


def get_session_id() -> str:
    return SESSION_ID


# ------ Constants ------

OPENAI_MODELS = {
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
}

TOGETHER_COMPUTE_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

TOKENIZER_MODEL_MAPS = {
    "mistralai/Mistral-7B-Instruct-v0.2": "gpt-4",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "gpt-4",
    "gpt-4-0125-preview": "gpt-4",
    "gpt-4-1106-preview": "gpt-4",
}

CONTEXT_LENGTHS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-1106-preview": 32768,
    "gpt-4-0125-preview": 32768,
    "gpt-4-vision-preview": 32768,
    "mistralai/Mistral-7B-Instruct-v0.2": 32768,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
}

DEFAULT_CONTEXT_LENGTH_FALLBACKS = {
    "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k": "gpt-4-1106-preview",
    "gpt-4": "gpt-4-0125-preview",
}

DEFAULT_FLAGSHIP_MODEL = "gpt-4"
DEFAULT_VISION_MODEL = "gpt-4-vision-preview"
DEFAULT_TOKENIZER_MODEL = "gpt-4"
DEFAULT_FLAGSHIP_LONG_CONTEXT_MODEL = "gpt-4-0125-preview"
DEFAULT_FAST_MODEL = "gpt-3.5-turbo"
DEFAULT_FAST_LONG_CONTEXT_MODEL = "gpt-3.5-turbo-16k"
CANDIDATE_FLAGSHIP_MODEL = "gpt-4-0125-preview"


# ------ Tokens ------


def encode(text, model_name):
    model_name = TOKENIZER_MODEL_MAPS.get(model_name, model_name)
    return tiktoken.encoding_for_model(model_name).encode(text)


def decode(tokens, model_name):
    model_name = TOKENIZER_MODEL_MAPS.get(model_name, model_name)
    return tiktoken.encoding_for_model(model_name).decode(tokens)


def count_tokens_in_str(text: str, model_name: str) -> int:
    assert isinstance(text, str)
    return len(encode(text, model_name))


def markdown_to_pdf(md_text: str, output_path: str, html_prefix: Optional[str] = None):
    html = mistune.html(md_text)
    if html_prefix is not None:
        html = html_prefix + html
    pdfkit.from_string(html, output_path)


def num_tokens_from_functions(
    functions: List[Dict[str, Any]], model_name: str = "gpt-3.5-turbo-0613"
):
    """Return the number of tokens used by a list of functions."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for function in functions:
        function_tokens = len(encoding.encode(function["name"]))
        function_tokens += len(encoding.encode(function["description"]))

        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                for propertiesKey in parameters["properties"]:
                    function_tokens += len(encoding.encode(propertiesKey))
                    v = parameters["properties"][propertiesKey]
                    for field in v:
                        if field == "type":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["type"]))
                        elif field == "description":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["description"]))
                        elif field == "enum":
                            function_tokens -= 3
                            for o in v["enum"]:
                                function_tokens += 3
                                function_tokens += len(encoding.encode(o))
                function_tokens += 11

        num_tokens += function_tokens

    num_tokens += 12
    return num_tokens


def clip_text(
    text: str, num_tokens: int, model_name: str, add_truncation_marker: bool = True
) -> str:
    tokens = encode(text, model_name)
    if len(tokens) <= num_tokens:
        return text
    if add_truncation_marker:
        truncation_marker = "[...]"
        num_tokens -= count_tokens_in_str(truncation_marker, model_name)
        return decode(tokens[:num_tokens], model_name) + " " + truncation_marker
    else:
        return decode(tokens[:num_tokens], model_name)


@contextmanager
def using_openai_credentials(
    api_key: Optional[str] = None, endpoint: Optional[str] = None
):
    old_api_key = openai.api_key
    old_endpoint = openai.api_base
    if api_key is not None:
        openai.api_key = api_key
    if endpoint is not None:
        openai.api_base = endpoint
    yield
    openai.api_key = old_api_key
    openai.api_base = old_endpoint


def get_endpoint_and_key_for_model(
    model_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    if model_name in OPENAI_MODELS:
        return None, None
    if model_name in TOGETHER_COMPUTE_MODELS:
        return (
            "https://api.together.xyz",
            get_key("together_compute_api_key"),
        )
    raise ValueError(f"Model {model_name} not found")


# ------ URLs and Paths ------


def url_to_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc or parsed_url.path  # netloc for http(s), path for others


def domain_to_url(domain):
    if not domain.startswith(("http://", "https://", "www.")):
        return urlunparse(("https", domain, "", "", "", "")).strip("/") + "/"
    elif domain.startswith("www."):
        return urlunparse(("https", domain[4:], "", "", "", "")).strip("/") + "/"
    else:
        return domain.strip("/") + "/"


def email_to_url(email: str) -> Optional[str]:
    email_services = [
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "icloud.com",
        "aol.com",
        "zoho.com",
        "protonmail.com",
        "yandex.com",
        "mail.com",
        "gmx.com",
        "hotmail.com",  # Adding Hotmail as it's still commonly used
        "live.com",  # Microsoft's Live.com is also a popular choice
        "fastmail.com",  # FastMail is known for its speed and privacy
        "inbox.com",  # Inbox.com is another option, though less popular
        "rediffmail.com",  # Rediffmail is used in some regions, especially in India
    ]
    domain = email.split("@")[1]
    if domain in email_services:
        return None
    else:
        return domain_to_url(domain)


def get_email_path(email: str) -> str:
    return email.replace("@", "__at__").replace(".", "__dot__")


# ------ Caching ------


_cache_instance = None


class CacheConfig:
    _use_cache = True

    @classmethod
    def enable_cache(cls):
        cls._use_cache = True

    @classmethod
    def disable_cache(cls):
        cls._use_cache = False

    @classmethod
    def is_cache_enabled(cls):
        return cls._use_cache


def get_cache():
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = Cache(get_path("data/cache"))
    return _cache_instance


def custom_memoize(expire=None, tag=None, use_cache=True):
    def decorator(func):
        cache = get_cache()

        @wraps(func)
        def wrapper(*args, **kwargs):
            if CacheConfig.is_cache_enabled():
                # If caching is enabled, use the memoized version of the function
                @cache.memoize(expire=expire, tag=tag)
                def cached_func(*a, **k):
                    return func(*a, **k)

                return cached_func(*args, **kwargs)
            else:
                # If caching is disabled, directly call the original function
                return func(*args, **kwargs)

        return wrapper

    return decorator


def safe_json_dump(data, filepath, allow_fallback=False):
    # This ensures we don't accidentally overwrite an existing file with a corrupt file.
    temp_filename = filepath + ".tmp"
    pickle_filename = filepath + ".pkl"

    try:
        # Serialize to a string
        serialized_data = json.dumps(data, indent=2)

        # Write to a temporary file
        with open(temp_filename, "w") as temp_file:
            temp_file.write(serialized_data)

        # Verify temporary file
        with open(temp_filename, "r") as temp_file:
            json.load(temp_file)

        # Use shutil.move for an atomic operation
        shutil.move(temp_filename, filepath)

    except (TypeError, json.JSONDecodeError) as json_error:
        if allow_fallback:
            # Fallback to pickle if JSON fails
            try:
                with open(pickle_filename, "wb") as pickle_file:
                    pickle.dump(data, pickle_file)
                return pickle_filename
            except Exception as pickle_error:
                raise Exception(
                    f"Pickle serialization failed: {pickle_error}"
                ) from pickle_error
        else:
            raise Exception(f"JSON serialization failed: {json_error}") from json_error

    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return filepath


# ------ Numpy ------


def to_normed_array(x: Union[List[float], List[List[float]]]) -> np.ndarray:
    x = np.array(x)
    if x.ndim == 2:
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    else:
        return x / np.linalg.norm(x)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    min_score = scores.min()
    score_range = scores.max() - min_score
    if score_range == 0:
        return np.ones_like(scores)
    else:
        return (scores - min_score) / score_range


def find_outlier_threshold(scores: np.ndarray) -> float:
    # Find the histogram of scores
    hist, bin_edges = np.histogram(scores.flatten(), bins=200)
    # Define the algo
    algo = rpt.Pelt(model="l2", min_size=1).fit(hist)
    # Find the change points
    result = algo.predict(pen=10 * hist.mean())
    # Find the last change point
    last_change_point = result[-2]
    # Find the threshold
    threshold = bin_edges[last_change_point]
    return threshold
