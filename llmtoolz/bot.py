import json
import os
import re
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, fields
import time
from datetime import datetime

import tiktoken
import yaml
from common.logger import logger
from typing import Optional, List, Dict, Callable, Tuple, Any, Union, Mapping
import backoff as backoff
from hashlib import sha256
import openai
from openai_functions import FunctionWrapper

from .bot_utils import OutOfTokenCapError
from .utils import (
    custom_memoize as memoize,
    clip_text,
    parse_prompts,
    get_session_id,
    CONTEXT_LENGTHS,
    DEFAULT_CONTEXT_LENGTH_FALLBACKS,
    find_prompt_path,
    count_tokens_in_str,
    num_tokens_from_functions,
    find_markdown_header,
    get_endpoint_and_key_for_model,
    using_openai_credentials,
    DEFAULT_FLAGSHIP_MODEL,
    image_to_base64,
    find_text_under_header,
    TimeoutException,
    format_string,
    get_path
)



@dataclass
class Event:
    role: str
    content: Optional[str]
    tag: Optional[str] = None
    timestamp: Optional[float] = None
    # This is for function calls (only valid when role is "assistant")
    function_call: Optional[Dict[str, str]] = None
    # This is for function returns (only valid when role is "function")
    name: Optional[str] = None
    # This is for images
    image_url: Optional[str] = None
    image_detail: str = "auto"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.function_call is not None:
            assert (
                self.role == "assistant"
            ), f"Function call {self.function_call} is not valid for role {self.role}"

            def sanitize_name(name: str) -> str:
                # Replace invalid characters with '_'
                sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
                # Ensure the name is at least 1 character long and up to 64 characters
                return sanitized[:64] if sanitized else "_"

            self.function_call["name"] = sanitize_name(self.function_call["name"])

        if self.name is not None:
            assert (
                self.role == "function"
            ), f"Function name {self.name} is not valid for role {self.role}"

        if self.image_url is not None:
            # If the image is local, we'll need to load it and encode it as base64
            if os.path.exists(self.image_url):
                self.image_url = image_to_base64(self.image_url)

    @classmethod
    def from_mapping(cls, mapping: Mapping):
        # Mapping might have extra keys that we don't need
        valid_field_names = [f.name for f in fields(cls)]
        return cls(**{k: v for k, v in mapping.items() if k in valid_field_names})

    def get_num_tokens(self, model_name: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model_name=model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = tokens_per_message
        if self.function_call is not None:
            num_tokens += len(encoding.encode(self.function_call["name"]))
            num_tokens += len(encoding.encode(self.function_call["arguments"]))
        if self.content is not None:
            num_tokens += len(encoding.encode(self.content))
        if self.image_url is not None:
            if self.image_detail == "low":
                num_tokens += 65
            elif self.image_detail in ["auto", "high"]:
                # It's actually 129 per crop, but we'll just say 129
                num_tokens += 129
        if self.name is not None:
            num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)

    def __contains__(self, item):
        return item in self.__dict__

    def __str__(self):
        event_str = f"{self.role.upper()}: {self.content}"
        if self.image_url is not None:
            event_str = f"{event_str} [+ IMAGE, Detail {self.image_detail}]"
        if self.function_call is not None:
            event_str = f"{event_str} [FUNCTION CALL: {self.function_call['name']}({self.function_call['arguments']})]"
        if self.name is not None:
            event_str = f"{event_str} [FUNCTION: {self.name}]"
        return event_str

    def parse_function_call_name_and_argument(
        self,
    ) -> Union[Tuple[str, Dict[str, str]], Tuple[None, None]]:
        if self["role"] != "assistant":
            return None, None
        if self.function_call is None:
            return None, None
        name = self["function_call"]["name"]
        arguments = self["function_call"]["arguments"]
        arguments = json.loads(arguments)
        return name, arguments

    def state_dict(self):
        return self.__dict__

    def find_markdown_header_in_content(self, header: str) -> bool:
        if self.content is None:
            return False
        if not header.startswith("#"):
            header = f"# {header}"
        return find_markdown_header(self.content, header)

    def find_content_under_header(self, header: str) -> Optional[str]:
        if self.content is None:
            return None
        if not header.startswith("#"):
            header = f"# {header}"
        return find_text_under_header(self.content, header)

    def load_state_dict(self, state_dict: dict):
        self.role = state_dict["role"]
        self.content = state_dict["content"]
        self.tag = state_dict["tag"]
        self.timestamp = state_dict["timestamp"]
        self.image_url = state_dict.get("image_url", None)
        self.image_detail = state_dict.get("image_detail", "auto")
        # This is to ensure that we can load old events in a consistent way
        if self.role == "assistant":
            self.function_call = state_dict["function_call"]
        else:
            self.function_call = None
        if self.role == "function":
            self.name = state_dict["name"]
        else:
            self.name = None
        return self

    def render(self) -> Dict[str, str]:
        kwargs = {}
        if self.role == "assistant" and self.function_call is not None:
            # this is for backwards compatibility
            kwargs["function_call"] = self.function_call
        if self.role == "function":
            kwargs["name"] = self.name
        # Create the content
        if self.content is not None and self.image_url is None:
            # This is the normal case, where we have text
            content = self.content
        elif self.content is not None and self.image_url is not None:
            # In this case, we have both text and an image
            content = [
                dict(type="text", text=self.content),
                dict(
                    type="image_url",
                    image_url=dict(url=self.image_url, detail=self.image_detail),
                ),
            ]
        elif self.content is None and self.image_url is not None:
            # In this case, we only have an image and no text to go with it
            content = [
                dict(
                    type="image_url",
                    image_url=dict(url=self.image_url, detail=self.image_detail),
                )
            ]
        else:
            # This is the case where we have no text and no image
            # (maybe just function call)
            content = None
        return dict(role=self.role, content=content, **kwargs)

    def get_content_hash(self, exclude_timestamp: bool = False) -> str:
        content = [
            str(self.content),
            str(self.role),
            str(self.tag),
        ]
        if not exclude_timestamp:
            content.append(str(self.timestamp))
        if self.function_call is not None:
            content.append(str(self.function_call["name"]))
            content.append(str(self.function_call["arguments"]))
        else:
            content.append("None")
        if self.name is not None:
            content.append(str(self.name))
        else:
            content.append("None")
        if self.image_url is not None:
            content.append(str(self.image_url))
        else:
            content.append("None")
        if self.image_detail is not None:
            content.append(str(self.image_detail))
        else:
            content.append("None")
        return sha256("+".join(content).encode("utf-8")).hexdigest()


@dataclass
class EventContainer:
    events: List[Event] = field(default_factory=list)

    def __len__(self):
        return len(self.events)

    def append(self, event: Union[Event, Dict[str, Any]]):
        if isinstance(event, dict):
            event = Event(**event)
        self.events.append(event)

    def __iter__(self):
        return iter(self.events)

    def __getitem__(self, index):
        return self.events[index]

    def __setitem__(self, index, value):
        self.events[index] = value

    def __delitem__(self, index):
        del self.events[index]

    def __contains__(self, item):
        return item in self.events

    def __reversed__(self):
        new_event = deepcopy(self)
        new_event.events = list(reversed(new_event.events))
        return new_event

    def delete_at_indices(self, indices: List[int]) -> "EventContainer":
        for index in sorted(indices, reverse=True):
            del self.events[index]
        return self

    def get_num_tokens(self, model_name: str):
        return sum(event.get_num_tokens(model_name) for event in self.events)

    def state_dict(self) -> dict:
        return dict(events=[event.state_dict() for event in self.events])

    def load_state_dict(self, state_dict: dict):
        self.events = []
        # TODO: Remove this once shit stops breaking in the convos on disk
        if "events" not in state_dict:
            state_dict = dict(events=state_dict)
        for state in state_dict["events"]:
            self.events.append(Event.from_mapping(state))
        return self

    def clone(self) -> "EventContainer":
        return deepcopy(self)

    def render(self) -> List[Dict[str, str]]:
        return [event.render() for event in self.events]

    def detect_duplicate_events(
        self,
        start_index: int = 0,
        stop_index: Optional[int] = None,
        step: int = 1,
        event_filter: Optional[Callable[[Event], bool]] = None,
    ) -> bool:
        events = self.events[start_index:stop_index:step]
        # Filter based on the filter function
        if event_filter is not None:
            events = [event for event in events if event_filter(event)]
        # Get the hashes
        hashes = [event.get_content_hash(exclude_timestamp=True) for event in events]
        # Check if there are duplicates
        return len(hashes) != len(set(hashes))


class HistoryProcessor:
    def process(self, container: EventContainer, **kwargs) -> EventContainer:
        raise NotImplementedError

    def __call__(self, container: EventContainer, **kwargs) -> EventContainer:
        return self.process(container, **kwargs)

    @staticmethod
    def apply_pipeline(
        history_processors: List["HistoryProcessor"],
        container: EventContainer,
        **kwargs,
    ) -> EventContainer:
        for processor in history_processors:
            container = processor(container, **kwargs)
        return container


class EventTruncater(HistoryProcessor):
    def __init__(
        self,
        model_name: str = DEFAULT_FLAGSHIP_MODEL,
        context_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        if context_length is not None:
            self.context_length = context_length
        else:
            self.context_length = CONTEXT_LENGTHS.get(self.model_name, 4096)

    def is_noop(self, total_tokens: int) -> bool:
        if total_tokens < self.context_length:
            return True
        return False

    def process(self, container: EventContainer, **kwargs) -> EventContainer:
        raise NotImplementedError


class ResponseProcessor:
    def __init__(self, fn: callable, one_time_use: bool = False, static: bool = True):
        # Public
        self.fn = fn
        self.one_time_use = one_time_use
        self.static = static
        # Privates
        self._is_used = False

    @property
    def is_used(self):
        if self._is_used and self.one_time_use:
            return True
        return False

    def __call__(self, event: Event, **kwargs) -> Event:
        if self.is_used:
            return event
        self._is_used = True
        if self.static:
            return self.fn(event, **kwargs)
        else:
            return self.fn(self, event, **kwargs)

    @staticmethod
    def apply_pipeline(
        response_processors: List["ResponseProcessor"],
        event: Event,
        remove_used_processors_in_place: bool = True,
        **kwargs,
    ) -> Event:
        # Remove the used processors
        if remove_used_processors_in_place:
            # Loop over the processors and delete the ones that are used
            response_processors[:] = [
                processor for processor in response_processors if not processor.is_used
            ]
        else:
            # Loop over the processors and delete the ones that are used
            response_processors = [
                processor for processor in response_processors if not processor.is_used
            ]

        for processor in response_processors:
            event = processor(event, **kwargs)
        return event


class ChatHistory:
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        history_processors: List["HistoryProcessor"] = None,
        response_processors: List["ResponseProcessor"] = None,
    ):
        # Private
        self._container = EventContainer()
        # Public
        self.history_processors = history_processors or []
        self.response_processors = response_processors or []
        # Logics
        if system_prompt is not None:
            self.add_system_event(system_prompt)

    @staticmethod
    def create_api_event(
        role: str, content: str, tag: Optional[str] = None, **kwargs
    ) -> Event:
        event = Event(role=role, content=str(content), tag=tag, **kwargs)
        return event

    @property
    def system_prompt(self) -> Optional[str]:
        for event in self._container:
            if event["role"] == "system":
                return event["content"]

    def update_system_prompt(self, prompt: str) -> None:
        for event in self._container:
            if event["role"] == "system":
                event["content"] = prompt
                return

    @property
    def events(self) -> EventContainer:
        return self._container.clone()

    def register_history_processor(self, history_processor: HistoryProcessor):
        self.history_processors.append(history_processor)
        return self

    def register_response_processor(self, response_processor: ResponseProcessor):
        self.response_processors.append(response_processor)
        return self

    def add_event(self, content: str, role: str, **kwargs) -> "ChatHistory":
        self._container.append(self.create_api_event(role, content, **kwargs))
        return self

    def add_system_event(self, content: str, **kwargs) -> "ChatHistory":
        self.add_event(content, role="system", **kwargs)
        return self

    def add_user_event(self, content: str, **kwargs) -> "ChatHistory":
        self.add_event(content, role="user", **kwargs)
        return self

    def add_function_event(
        self, function_name: str, function_result: Any
    ) -> "ChatHistory":
        self.add_event(
            json.dumps(function_result, indent=2),
            role="function",
            name=function_name,
        )
        return self

    def record_response(self, event: Mapping) -> "Event":
        assert "role" in event
        assert "content" in event
        event = Event.from_mapping(event)
        # Process the event
        event = ResponseProcessor.apply_pipeline(
            self.response_processors, event, remove_used_processors_in_place=True
        )
        self._container.append(event)
        return event

    def get_most_recent_assistant_event(self) -> Optional[Event]:
        for event in reversed(self._container):
            if event["role"] == "assistant":
                return event
        return None

    def get_most_recent_assistant_event_by_content_match(
        self, content_to_match: str
    ) -> Optional[Event]:
        for event in reversed(self._container):
            if event["role"] == "assistant" and content_to_match in event["content"]:
                return event
        return None

    def get_most_recent_event(self) -> Optional[Event]:
        if len(self._container) == 0:
            return None
        return self._container[-1]

    def render(
        self,
        system_prompt_format_kwargs: Optional[dict] = None,
        num_additional_tokens: int = 1024,
        render_container: bool = True,
    ) -> Union[List[Dict[str, Any]], EventContainer]:
        # Format with system prompt. This needs to happen first because it
        # adds tokens.
        container = deepcopy(self._container)
        if system_prompt_format_kwargs is not None:
            for event in container:
                if event["role"] == "system":
                    event["content"] = format_string(
                        event["content"], **system_prompt_format_kwargs
                    )
        # Process the container

        container = HistoryProcessor.apply_pipeline(
            self.history_processors,
            container=container,
            num_additional_tokens=num_additional_tokens,
        )
        # Render the container
        if render_container:
            return container.render()
        else:
            return container

    def state_dict(self) -> dict:
        return dict(container=self._container.state_dict())

    def load_state_dict(self, state_dict: dict) -> "ChatHistory":
        self._container.load_state_dict(state_dict["container"])
        return self

    def __str__(self):
        lines = []
        for event in self._container:
            if event["role"] in ["system", "user"]:
                lines.append(f"{event['role'].upper()}: {event['content']}")
            elif event["role"] == "assistant":
                lines.append(f"{event['role'].upper()}: {event['content']}")
                if "function_call" in event:
                    lines.append(
                        f"FUNCTION CALL: {event['function_call']['name']}({event['function_call']['arguments']})"
                    )
            elif event["role"] == "function":
                lines.append(f"FUNCTION: {event['name']}(...) = {event['content']}")
        return "\n------\n".join(lines)

    def handle_overflow(
        self,
        model_name: str,
        context_length: int,
        max_tokens_to_generate: int,
        buffer_size: int = 500,
    ):
        # TODO: this is a hacky way to handle overflow, it assumes that the last content contains sufficient tokens to be truncated
        total_tokens = (
            self._container.get_num_tokens(model_name)
            + max_tokens_to_generate
            + buffer_size
        )

        while total_tokens > context_length:
            logger.warning(
                f"Handling overflow (total_tokens={total_tokens}, context_length={context_length})"
            )
            # remove the last half of the latest content
            last_message = self._container[-1]["content"]
            num_tokens_in_last_message = count_tokens_in_str(last_message, model_name)
            target_num_tokens_in_last_message = round(num_tokens_in_last_message * 0.9)
            self._container[-1]["content"] = clip_text(
                last_message, target_num_tokens_in_last_message, model_name
            )
            total_tokens = (
                self._container.get_num_tokens(model_name)
                + max_tokens_to_generate
                + buffer_size
            )

        return self


class Bot:
    AUTO_PERSIST_BY_DEFAULT = False
    VERSION_PREFERENCES = {}
    TOKEN_CAP = 32768

    def __init__(
        self,
        name,
        model_name,
        *,
        system_prompt: Optional[str] = None,
        context_length: Optional[int] = None,
        fallback_when_out_of_context: bool = False,
        out_of_context_fallback_model_name: Optional[str] = None,
        auto_persist: Optional[bool] = None,
        system_prompt_format_kwargs: Optional[dict] = None,
        token_cap: Optional[int] = None,
        prompt_version: Optional[str] = None,
    ):
        # Get the names
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.name = f"{name}__{model_name}__{get_session_id()}__{timestamp}__{str(uuid.uuid4().hex[:5])}"
        self.model_name = model_name
        if out_of_context_fallback_model_name is None and fallback_when_out_of_context:
            out_of_context_fallback_model_name = DEFAULT_CONTEXT_LENGTH_FALLBACKS.get(
                model_name
            )
        self.out_of_context_fallback_model_name = out_of_context_fallback_model_name

        # Logging
        if auto_persist is None:
            auto_persist = self.AUTO_PERSIST_BY_DEFAULT
        self.auto_persist = auto_persist

        # History business
        self.user_prompt = None
        if context_length is None:
            self.context_length = CONTEXT_LENGTHS.get(model_name, 4096)
        else:
            self.context_length = context_length
        if token_cap is None:
            self.token_cap = self.TOKEN_CAP
        else:
            self.token_cap = token_cap
        # Try to load the system prompt from disk if it's not provided
        if system_prompt is None:
            try:
                prompt_version = prompt_version or self.VERSION_PREFERENCES.get(
                    name, None
                )
                system_prompt, user_prompt = parse_prompts(
                    find_prompt_path(name, prompt_version)
                )
                self.user_prompt = user_prompt
            except FileNotFoundError:
                logger.warning(
                    "Initializing bot without system prompt. This is OK if loading a bot from history."
                )
        if system_prompt_format_kwargs is not None:
            system_prompt = format_string(system_prompt, **system_prompt_format_kwargs)
        self.history = ChatHistory(system_prompt=system_prompt)
        self._functions = {}
        self._enabled_functions = set()

    @property
    def system_prompt(self):
        return self.history.system_prompt

    def format_user_prompt(self, **kwargs) -> str:
        if self.user_prompt is None:
            raise ValueError("User prompt not set.")
        return format_string(self.user_prompt, **kwargs)

    @classmethod
    def register_version_preference(cls, name: str, version_string: str) -> type:
        cls.VERSION_PREFERENCES[name] = version_string
        return cls

    def register_function(
        self,
        function: Callable[[Any], Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        enabled: bool = True,
    ) -> "Bot":
        if name is None:
            name = function.__name__
        self._functions[name] = FunctionWrapper(
            function, name=name, description=description
        )
        if enabled:
            self.enable_function(name)
        return self

    def enable_function(self, name: str) -> "Bot":
        self._enabled_functions.add(name)
        return self

    def enable_all_functions(self):
        self._enabled_functions = set(self._functions.keys())
        return self

    def disable_function(self, name: str) -> "Bot":
        self._enabled_functions.remove(name)
        return self

    def disable_all_functions(self):
        self._enabled_functions = set()
        return self

    def get_function(self, name: str) -> FunctionWrapper:
        return self._functions[name]

    @contextmanager
    def these_functions_enabled(self, functions: List[str]):
        enabled_functions = self._enabled_functions
        self._enabled_functions = set(functions)
        yield
        self._enabled_functions = enabled_functions

    def get_schema_for_enabled_functions(self):
        for name in self._enabled_functions:
            yield self._functions[name].schema

    def load_system_prompt(self, name: str, prompt_version: Optional[str] = None):
        prompt_version = prompt_version or self.VERSION_PREFERENCES.get(name, None)
        system_prompt, user_prompt = parse_prompts(
            find_prompt_path(name, prompt_version)
        )
        self.history.update_system_prompt(system_prompt)
        self.user_prompt = user_prompt
        return self

    def complete(
        self,
        max_tokens: int = 750,
        temperature: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        handle_overflow: bool = False,
        system_prompt_format_kwargs: Optional[dict] = None,
    ):
        functions = list(self.get_schema_for_enabled_functions())
        if len(functions) > 0:
            kwargs = dict(functions=functions)
        else:
            kwargs = dict()

        if stop is not None:
            if isinstance(stop, str):
                stop = [stop]
            kwargs["stop"] = stop

        @backoff.on_exception(
            backoff.expo,
            (
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.Timeout,
                openai.error.APIError,
                openai.error.APIConnectionError,
                TimeoutException,
            ),
            max_time=600,
        )
        @memoize(expire=3600, tag="bot")
        def create_chat_completion(
            model_name: str,
            rendered_history: List[Dict[str, Any]],
            temperature: float,
            max_tokens: int,
            **kwargs,
        ):
            try:
                endpoint, api_key = get_endpoint_and_key_for_model(model_name)
                with using_openai_credentials(api_key=api_key, endpoint=endpoint):
                    return_value = openai.ChatCompletion.create(
                        model=model_name,
                        messages=rendered_history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        request_timeout=600,
                        **kwargs,
                    )

            except Exception as e:
                self.log_api_call(
                    model_name=model_name,
                    rendered_history=rendered_history,
                    temperature=temperature,
                    bot_name=self.name,
                    return_value=dict(error=f"error: {e}"),
                )
                raise e

            path = self.log_api_call(
                model_name=model_name,
                rendered_history=rendered_history,
                temperature=temperature,
                bot_name=self.name,
                return_value=dict(return_value),
            )

            return return_value

        num_function_tokens = num_tokens_from_functions(functions, self.model_name)
        token_buffer_size = 25
        history_container = self.history.render(
            system_prompt_format_kwargs=system_prompt_format_kwargs,
            num_additional_tokens=max_tokens + num_function_tokens,
            render_container=False,
        )

        if self.out_of_context_fallback_model_name is not None:
            total_tokens = (
                history_container.get_num_tokens(self.model_name)
                + max_tokens
                + num_function_tokens
                + token_buffer_size
            )
            # Check if we'll run out of context
            if total_tokens > self.context_length:
                model_name = self.out_of_context_fallback_model_name
            else:
                model_name = self.model_name
        else:
            model_name = self.model_name

        # FIXME: Absorb this in history processor
        if handle_overflow:
            self.history.handle_overflow(
                model_name,
                context_length=CONTEXT_LENGTHS.get(model_name, 4096),
                max_tokens_to_generate=max_tokens,
            )

        rendered_history = history_container.render()
        num_history_tokens = history_container.get_num_tokens(model_name)
        total_possible_tokens = (
            num_history_tokens + max_tokens + num_function_tokens + 25
        )

        if total_possible_tokens > self.token_cap:
            raise OutOfTokenCapError(
                f"Total possible tokens ({total_possible_tokens}) exceeds token cap ({self.token_cap})."
            )

        logger.debug(
            f"Passing into {model_name} num_history_tokens: {num_history_tokens}, "
            f"num_generation_tokens: {max_tokens}, "
            f"num_function_tokens: {num_function_tokens}; "
            f"total_possible_tokens: {total_possible_tokens}"
        )

        try:
            completion = create_chat_completion(
                model_name,
                rendered_history,
                temperature,
                max_tokens,
                **kwargs,
            )
        except openai.error.InvalidRequestError as e:
            if "reduce the length of the messages" in str(e).lower():
                # Somehow we still managed to go out of context, probably due
                # to miscounted tokens. We'll try again with the fallback model
                if self.out_of_context_fallback_model_name is not None:
                    logger.warning(
                        f"Out of context error with {model_name}, "
                        f"trying fallback model {self.out_of_context_fallback_model_name}."
                    )
                    completion = create_chat_completion(
                        self.out_of_context_fallback_model_name,
                        rendered_history,
                        temperature,
                        max_tokens,
                        **kwargs,
                    )
                else:
                    logger.error(
                        "Out of context error, but no fallback model specified."
                    )
                    raise
            else:
                raise

        response = completion.choices[0].message

        # This might transform the response, depending on the attached processors
        response = self.history.record_response(response)

        if self.auto_persist:
            self.persist()

        return response

    def log_api_call(
        self,
        model_name: str,
        rendered_history: List[Dict[str, Any]],
        temperature: float,
        bot_name: str,
        return_value: dict,
    ):
        log_payload = {
            "model_name": model_name,
            "rendered_history": rendered_history,
            "temperature": temperature,
            "bot_name": bot_name,
            "return_value": return_value,
        }
        path = self.api_log_json_path
        with open(path, "w") as f:
            json.dump(log_payload, f, indent=2)
        return path

    def call_requested_function(self, add_to_history: bool = True) -> Any:
        event = self.history.get_most_recent_assistant_event()

        if event.function_call is None:
            # No function to call
            return None

        try:
            (name, arguments) = event.parse_function_call_name_and_argument()
            if name in self._functions:
                function = self._functions[name]
            else:
                raise KeyError(
                    f"Function with name '{name}' not found. "
                    f"Available functions are: {', '.join(self._functions.keys())}"
                )
            function_result = function(arguments)
        except Exception as e:
            logger.exception(f"Error calling function {name}")
            function_result = dict(error=str(e))

        if add_to_history:
            self.history.add_function_event(name, function_result)

            if self.auto_persist:
                self.persist()

        return function_result

    def enable_context_length_fallback(self):
        self.out_of_context_fallback_model_name = DEFAULT_CONTEXT_LENGTH_FALLBACKS.get(
            self.model_name
        )
        return self

    def state_dict(self) -> dict:
        return dict(
            name=self.name,
            model_name=self.model_name,
            history=self.history.state_dict(),
            out_of_context_fallback_model_name=self.out_of_context_fallback_model_name,
        )

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> "Bot":
        if strict:
            assert (
                self.model_name == state_dict["model_name"]
            ), f"Model name mismatch: {self.model_name} != {state_dict['model_name']}"
        self.history.load_state_dict(state_dict["history"])
        return self

    @property
    def persist_directory(self) -> str:
        log_name = self.name.replace(" ", "_").replace(".", "-").replace("/", "-")
        return get_path(f"data/logs/bots/{get_session_id()}/{log_name}", makedirs=True)

    @property
    def api_log_json_path(self):
        log_name = self.name.replace(" ", "_").replace(".", "-").replace("/", "-")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return get_path(
            relative_path=f"data/logs/api_calls/{get_session_id()}/{log_name}/{timestamp}.json",
            makedirs=True,
        )

    def persist(self):
        # 1. List all files in the specified directory
        files = os.listdir(self.persist_directory)

        # 2. Filter out files that follow the pattern `history_v{version}.json` and extract version numbers
        version_numbers = [
            int(match.group(1))
            for file in files
            if (match := re.match(r"history_v(\d+)\.json", file))
        ]

        # 3. Find the maximum version number
        max_version = max(version_numbers, default=0)

        # 4. Increment the maximum version number by 1 to get the new version number
        new_version = max_version + 1

        # 5. Save self.state_dict() as a JSON file
        file_path = os.path.join(self.persist_directory, f"history_v{new_version}.json")
        with open(file_path, "w") as f:
            json.dump(self.state_dict(), f, indent=2)

        logger.debug(f"State dict saved to {file_path}")

    @staticmethod
    def find_latest_history_file(directory: str) -> str:
        # List all files in the directory
        files = os.listdir(directory)

        # Filter out files that follow the pattern `history_v{version}.json` and extract version numbers
        versioned_files = [
            (file, int(match.group(1)))
            for file in files
            if (match := re.match(r"history_v(\d+)\.json", file))
        ]

        # Find the file with the maximum version number
        if not versioned_files:
            raise FileNotFoundError(
                "No history files found in the specified directory."
            )

        latest_file, _ = max(versioned_files, key=lambda x: x[1])
        return os.path.join(directory, latest_file)

    def load_from_path(self, path: str) -> "Bot":
        if os.path.isdir(path):
            # Find the latest history file in the directory
            latest_file_path = self.find_latest_history_file(path)

            # Load the state dictionary from this file
            with open(latest_file_path, "r") as f:
                state_dict = json.load(f)

        elif os.path.isfile(path):
            # Load the state dictionary directly from the file
            with open(path, "r") as f:
                state_dict = json.load(f)

        else:
            raise ValueError(f"Invalid path: {path}")

        # Use the load_state_dict method to load the state dictionary into the object
        self.load_state_dict(state_dict)
        return self

    @classmethod
    def from_path(
        cls,
        path: str,
        bot_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> "Bot":
        if os.path.isdir(path):
            file = cls.find_latest_history_file(path)
        elif os.path.isfile(path):
            file = path
        else:
            raise ValueError(f"Invalid path: {path}")
        with open(file, "r") as f:
            state_dict = json.load(f)
        logger.debug(f"Loaded state dict from {file}.")
        return cls.from_dict(state_dict, bot_name=bot_name, model_name=model_name)

    @classmethod
    def from_dict(
        cls,
        state_dict: dict,
        bot_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> "Bot":
        if bot_name is None:
            bot_name = state_dict["name"].split("__")[0]

        if model_name is None:
            model_name = state_dict["model_name"]

        logger.debug(f"Creating bot {bot_name} with model {model_name}.")

        bot = cls(name=bot_name, model_name=model_name)
        bot.load_state_dict(state_dict)
        return bot


@dataclass
class GlobalBotConfig:
    version_pins: Dict[str, str] = field(default_factory=dict)
    auto_persist: bool = False

    def configure(self, bot: Optional["Bot"] = None):
        if bot is None:
            bot = Bot
        for name, version in self.version_pins.items():
            bot.register_version_preference(name, version)
        bot.AUTO_PERSIST_BY_DEFAULT = self.auto_persist
        return self

    @classmethod
    def from_dict(cls, d: Dict) -> "GlobalBotConfig":
        return cls(**d)

    @classmethod
    def from_path(cls, path: str) -> "GlobalBotConfig":
        if not os.path.exists(path):
            # This is a relative path
            path = get_path(path)
        assert os.path.exists(path), f"Path {path} does not exist"
        with open(path, "r") as f:
            d = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_dict(d)


@memoize(tag="embed")
def embed(
    text: str,
    model_name: str = "text-embedding-ada-002",
    clip_if_too_long: bool = True,
    **embedder_kwargs,
) -> List[float]:
    if clip_if_too_long:
        # TODO: Make this work for other embedding models
        # Clip text to a maximum of 8192 tokens
        text = clip_text(text, 8192, model_name)

    @backoff.on_exception(
        backoff.expo,
        (
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
            openai.error.APIError,
            TimeoutException,
        ),
        max_time=180,
    )
    def _get_embedding(_text):
        embedding = openai.Embedding.create(
            input=[_text], model=model_name, **embedder_kwargs
        )["data"][0]["embedding"]
        return embedding

    return _get_embedding(text)


@backoff.on_exception(
    backoff.expo,
    (
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        openai.error.Timeout,
        openai.error.APIError,
        TimeoutException,
    ),
    max_time=180,
)
def batch_embed(
    texts: List[str],
    model_name: str = "text-embedding-3-small",
    clip_if_too_long: bool = True,
    **kwargs,
) -> List[List[float]]:
    if clip_if_too_long:
        # Clip text to a maximum of 8192 tokens
        texts = [clip_text(text, 8192, model_name) for text in texts]
    embeddings = openai.Embedding.create(input=texts, model=model_name, **kwargs)
    embeddings = [embedding["embedding"] for embedding in embeddings["data"]]
    return embeddings
