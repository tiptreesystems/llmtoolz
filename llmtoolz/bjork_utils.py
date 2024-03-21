import contextlib
import json
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import Optional, Tuple, List, Union, Any

from common.logger import logger

from .bot import Bot
from .multitool import ToolDefinition, ToolDataType
from .scrape import WebPage
from .search import GoogleSearch
from .utils import (
    CONTEXT_LENGTHS,
    find_text_under_header,
    clip_text,
    DEFAULT_FLAGSHIP_MODEL,
    DEFAULT_FAST_MODEL,
    parse_json,
)

FAILED_TO_ACCESS_DOCUMENTS = "Failed to download any of the selected documents. Please try another search strategy."
FAILED_TO_SELECT_DOCUMENTS = (
    "The search did not return useful results. Please try another search strategy."
)
NO_DOCUMENTS_AVAILABLE = "No documents available to query from."
FAILED_TO_QUERY_COMPANY_DATABASE = "Failed to query company database."

RETRIEVE_ANSWER_FORMAT_STRING = (
    "Information potentially relevant to the question: {answer}"
)
RETRIEVE_NON_ANSWER_FORMAT_STRING = (
    "Relevant information not found. Here is what was found instead: {answer}"
)

PERSON_NOT_FOUND = "Person not found."
DESCRIPTION_NOT_PARSEABLE = "Failed to parse provided description to a lookup query."


class SearchAndSummarize(ToolDefinition):
    TOOL_DOC = """Use this tool if you want to search Google and summarize the top few results based on a query that you provide it. You may use this tool in any language that you want."""

    ARGUMENT_DOC = """The argument `search_query` specifies what should be searched for on Google. The argument `search_intent`  is one or more sentences that specifies what exactly you are hoping to find in the search results. If you encountered any ambiguities while searching, you should specify them in the `search_intent` argument. The more information you provide, the better the results will be."""

    ARGUMENTS = {
        "search_intent": ToolDataType.STRING,
        "search_query": ToolDataType.STRING,
    }

    DISCLAIMER = """The information gathered by this tool is based on data available on the internet. Please be extra vigilant about any potential inconsistencies or inaccuracies in the information. If you see any red flags in the response, you must be sure to flag them in order to trigger further research."""

    def __init__(
        self,
        num_pages_to_select: int = 3,
        link_selection_model: str = DEFAULT_FLAGSHIP_MODEL,
        link_synthesis_model: str = DEFAULT_FLAGSHIP_MODEL,
        summarizer_model: str = DEFAULT_FLAGSHIP_MODEL,
        parallel: bool = True,
        num_threads: int = 3,
        pull_and_ask: bool = True,
        reason_about_documents_to_select: bool = True,
        include_disclaimer: bool = True,
        embedding_rerank_weight: Optional[float] = 0.8,
        considered_web_pages_: Optional[List["WebPage"]] = None,
    ):
        super().__init__(
            tool_name="search_and_summarize",
            function_name="execute",
            tool_doc=self.TOOL_DOC,
            argument_doc=self.ARGUMENT_DOC,
            arguments=self.ARGUMENTS,
        )
        self.num_pages_to_select = num_pages_to_select
        self.link_selection_model = link_selection_model
        self.link_synthesis_model = link_synthesis_model
        self.summarizer_model = summarizer_model
        self.parallel = parallel
        self.num_threads = num_threads
        self.pull_and_ask = pull_and_ask
        self.reason_about_documents_to_select = reason_about_documents_to_select
        self.include_disclaimer = include_disclaimer
        self.embedding_rerank_weight = embedding_rerank_weight
        self.considered_web_pages_ = considered_web_pages_

    def tool_fn(self, arguments: dict) -> Tuple[str, bool]:
        search_query = arguments["search_query"]
        search_intent = arguments["search_intent"]

        num_results = (self.num_pages_to_select * 2) + 1

        logger.user(f"Searching the web for: {search_query}")
        search = GoogleSearch.from_query(
            query=search_query,
            num_results=(
                100 if self.embedding_rerank_weight is not None else num_results
            ),
        )

        if self.embedding_rerank_weight is not None:
            logger.debug(
                f"Reranking search results (weight: {self.embedding_rerank_weight}) "
                f"based on search intent.",
            )
            search = search.rerank(
                query=search_intent, weight=self.embedding_rerank_weight
            ).keep_top_results(num_results=num_results)

        answer, was_successful = query_web_documents(
            query=search_intent,
            documents=search.results,
            max_links=self.num_pages_to_select,
            search_model=self.link_selection_model,
            synthesis_model=self.link_synthesis_model,
            summarizer_model=self.summarizer_model,
            include_urls=True,
            parallel=self.parallel,
            num_threads=self.num_threads,
            pull_and_ask=self.pull_and_ask,
            reason_about_documents_to_select=self.reason_about_documents_to_select,
            considered_web_pages_=self.considered_web_pages_,
        )

        if self.include_disclaimer:
            answer += f"\n\nIMPORTANT NOTE: {self.DISCLAIMER}"

        return answer, was_successful

class QueryWebPage(ToolDefinition):
    TOOL_DOC = """Use this tool if you want to scrape a web page and query its contents, given its URL."""

    ARGUMENT_DOC = """The argument `url` specifies the URL of the web page that you want to query. The URL can point to regular HTML web pages as well as PDFs. The argument `query` specifies the question that you want to ask the web page. Be as specific as possible, and provide as much context as you can."""

    ARGUMENTS = {
        "url": ToolDataType.STRING,
        "query": ToolDataType.STRING,
    }

    def __init__(self, summarizer_model: str = DEFAULT_FLAGSHIP_MODEL):
        super().__init__(
            tool_name="query_web_page",
            function_name="execute",
            tool_doc=self.TOOL_DOC,
            argument_doc=self.ARGUMENT_DOC,
            arguments=self.ARGUMENTS,
        )
        self.summarizer_model = summarizer_model

    def tool_fn(self, arguments: dict) -> Any:
        url = arguments["url"]
        query = arguments["query"]

        web_page = WebPage.from_url(url)
        answer, was_successful = query_web_documents(
            query=query,
            documents=[web_page],
            summarizer_model=self.summarizer_model,
            reason_about_documents_to_select=False,
            pull_and_ask=True,
            parallel=False,
        )
        return answer, was_successful


class PeopleLookup(ToolDefinition):
    TOOL_DOC = """Use this tool if you want to look up information about a person, such as their contact information and professional role. This tool is especially useful to get contact information like email addresses."""

    ARGUMENT_DOC = """The argument `description` should contain a description of the person that you are looking for. It should including their name, and any information you already have about e.g. where they work, etc. The more information you provide in this argument, the better. The argument `find_contact_details` should be set to `true` if you are looking for contact details, and `false` if you are not. Note that it costs money to get contact details, so only set this to `true` if you are sure that you need them."""

    ARGUMENTS = {
        "description": ToolDataType.STRING,
        "find_contact_details": ToolDataType.BOOLEAN,
    }

    def __init__(self, model_name: str = DEFAULT_FLAGSHIP_MODEL):
        super().__init__(
            tool_name="people_lookup",
            function_name="execute",
            tool_doc=self.TOOL_DOC,
            argument_doc=self.ARGUMENT_DOC,
            arguments=self.ARGUMENTS,
        )
        self.model_name = model_name

    def tool_fn(self, arguments: dict) -> Tuple[str, bool]:
        return people_lookup(
            description=arguments["description"],
            find_contact_details=arguments["find_contact_details"],
            model_name=self.model_name,
        )


def query_web_documents(
    query: str,
    documents: List["WebPage"],
    max_links: int = 3,
    search_model: str = DEFAULT_FAST_MODEL,
    synthesis_model: str = DEFAULT_FAST_MODEL,
    summarizer_model: str = DEFAULT_FLAGSHIP_MODEL,
    include_urls: bool = False,
    parallel: bool = True,
    num_threads: int = 3,
    pull_and_ask: bool = True,
    reason_about_documents_to_select: bool = True,
    considered_web_pages_: List["WebPage"] = None,
) -> Tuple[str, bool]:
    # Validate
    if len(documents) == 0:
        return NO_DOCUMENTS_AVAILABLE, False

    logger.debug(f"Selecting amongst {len(documents)} documents.")
    # Select the documents that we need to answer the query
    if reason_about_documents_to_select:
        selected_docs = select_relevant_documents(
            query=query,
            documents=documents,
            max_links=max_links,
            model_name=search_model,
        )
        logger.debug(
            f"Selected {len(selected_docs)} documents out of {len(documents)} for query: {query}.",
        )
    else:
        selected_docs = list(documents)

    if len(selected_docs) == 0:
        return FAILED_TO_SELECT_DOCUMENTS, False

    if pull_and_ask:
        # Pull the documents
        filtered_docs = pull_documents(
            documents=selected_docs,
            remove_unpullable_documents=True,
            parallel=parallel,
            num_threads=num_threads,
        )
    else:
        # If we're not allowed to pull the documents,
        # we filter out the ones without a description.
        filtered_docs = [doc for doc in selected_docs if doc.description is not None]

    if len(filtered_docs) == 0:
        return FAILED_TO_ACCESS_DOCUMENTS, False

    if pull_and_ask:
        # What this does is figures out the number of tokens that we can afford
        # to spend on each article before truncating it. It's a function of the
        # number of articles we have and the context length of the model.
        num_buffer_tokens = min(2048, int(CONTEXT_LENGTHS[synthesis_model] * 0.25))
        num_tokens_per_article = (
            CONTEXT_LENGTHS[synthesis_model] - num_buffer_tokens
        ) / len(filtered_docs)
        num_tokens_per_article = int(num_tokens_per_article)

        # ask_documents_in_bulk will look for the answer to the query in each doc, and
        # return that. If there is no answer, it will return None for that doc.
        answers = ask_documents_in_bulk(
            query=query,
            documents=filtered_docs,
            model_name=summarizer_model,
            parallel=parallel,
            num_threads=num_threads,
            clip_to_num_tokens=num_tokens_per_article,
        )
    else:
        answers = [
            format_with_metadata(content=f"Snippet: {doc.description}", document=doc)
            for doc in filtered_docs
        ]

    logger.debug(f"Answers: {answers}")
    logger.user("Summarizing answers...")
    # For the answers, we want to filter out the ones that are None
    # because they failed to summarize and don't have the answer in them.
    filtered_docs = [
        doc for doc, answer in zip(filtered_docs, answers) if answer is not None
    ]
    filtered_urls = [doc.url for doc in filtered_docs]
    selected_content = [answer for answer in answers if answer is not None]

    if len(selected_content) == 0:
        return FAILED_TO_ACCESS_DOCUMENTS, False

    # It's possible that we have no content left after filtering out the ones
    # that failed to summarize. In that case, we just say that no urls were found
    selected_content = "\n\n".join(selected_content)
    logger.debug(f"Selected content: {selected_content}")
    bot = Bot(
        name="bjork_summarizer",
        model_name=synthesis_model,
        fallback_when_out_of_context=True,
    )
    user_prompt = bot.format_user_prompt(query=query, selected_content=selected_content)
    bot.history.add_user_event(user_prompt)
    logger.debug(bot.system_prompt)
    logger.debug(user_prompt)
    bot_output = response = bot.complete(
        max_tokens=1536, temperature=0.1, handle_overflow=True
    )["content"]
    logger.debug(response)

    # Find the content
    response = find_text_under_header(response, header="# Answer", keep_subheaders=True)

    # If we don't have a response, let's try again
    if response is None:
        logger.warning(
            f"Failed to produce answer. Last bot output was: \n\n"
            f"{bot_output}. \n\n"
            f"Retrying.",
        )
        bot.history.add_user_event(
            "# System Information\n\n"
            'Please produce an answer under the markdown header "# Answer".'
        )
        bot_output = response = bot.complete(
            max_tokens=1536, temperature=0.1, handle_overflow=True
        )["content"]
        logger.debug(response)
        response = find_text_under_header(
            response, header="# Answer", keep_subheaders=True
        )

    # Get the summary line
    if response is None:
        logger.error(
            f"No summary or answer found. Last bot output:\n\n"
            f"{bot_output}\n\nReturning.",
        )
        logger.user("Failed to find the answer in the documents.")
        return FAILED_TO_ACCESS_DOCUMENTS, False

    logger.user(f"Found answer in the documents: {response}")

    if include_urls:
        urls = filtered_urls
        response = response + "\n\n## Sources:\n\n" + "\n".join(urls)

    if considered_web_pages_ is not None:
        considered_web_pages_.extend(filtered_docs)

    return response, True


def select_relevant_documents(
    query: str,
    documents: List[WebPage],
    max_links: int,
    model_name: str = DEFAULT_FLAGSHIP_MODEL,
) -> List[WebPage]:
    logger.debug(f"Selecting relevant documents for query: {query}.")
    logger.user("Thinking about which documents to select...")
    bot = Bot(name="bjork_url_selector", model_name=model_name)
    metadata = json.dumps([x.get_metadata() for x in documents], indent=2)
    user_prompt = bot.format_user_prompt(
        query=query, max_links=max_links, metadata=metadata
    )
    bot.history.add_user_event(user_prompt)
    logger.debug(bot.system_prompt)
    logger.debug(user_prompt)
    response = bot.complete(max_tokens=1024, temperature=0.1)["content"]
    logger.debug(response)
    selected_urls = [
        l.replace("URL:", "").strip()
        for l in response.split("\n")
        if l.startswith("URL:")
    ][:max_links]
    # Select the URLs
    selected_docs = [doc for doc in documents if doc.url in selected_urls]
    # User prints the selected URLs
    logger.user(find_text_under_header(response, "# Thoughts"))
    return selected_docs


def pull_documents(
    documents: List[WebPage],
    remove_unpullable_documents: bool = True,
    parallel: bool = True,
    num_threads: int = 3,
) -> List[WebPage]:
    logger.debug(f"Pulling documents {len(documents)} documents.")
    logger.user(f"Downloading {len(documents)} documents...")

    def pull_doc(doc):
        if doc.main_content is None or doc.main_content == "" or doc.pull_failed:
            return doc, False
        else:
            return doc, True

    def build_executor():
        if parallel and num_threads >= 1:
            executor = ThreadPoolExecutor(num_threads)
            map_fn = executor.map
        else:
            executor = contextlib.nullcontext()
            map_fn = map
        return executor, map_fn

    executor, map_fn = build_executor()

    with executor:
        if remove_unpullable_documents:
            filtered_docs = [
                doc
                for doc, pull_successful in map_fn(pull_doc, documents)
                if pull_successful
            ]
        else:
            filtered_docs = [
                doc for doc, pull_successful in map_fn(pull_doc, documents)
            ]

    return filtered_docs


def format_with_metadata(content: str, document: WebPage) -> str:
    metadata_str = "\n".join(
        [
            f"{key.title()}: {value}"
            for key, value in document.get_metadata(include_description=False).items()
        ]
    )
    return metadata_str + "\n\n" + content + "\n\n" + "---"


def ask_documents_in_bulk(
    query: str,
    documents: List[WebPage],
    model_name: str = DEFAULT_FLAGSHIP_MODEL,
    parallel: bool = True,
    num_threads: int = 3,
    clip_to_num_tokens: Optional[int] = None,
) -> List[Union[str, None]]:
    logger.debug(f"Asking the {len(documents)} documents the query: {query}")
    logger.user(f"Working on: {query}")

    def prepare_for_summary(doc: WebPage) -> Optional[str]:
        # Summarize the web page if requested
        try:
            answer_in_doc = doc.ask(
                question=query,
                model_name=model_name,
                answer_format_string=RETRIEVE_ANSWER_FORMAT_STRING,
                non_answer_format_string=RETRIEVE_NON_ANSWER_FORMAT_STRING,
            )
        except Exception as e:
            logger.exception(
                f"Failed to ask document. Exception: {str(e)}. Traceback follows."
            )
            answer_in_doc = None

        if answer_in_doc is not None:
            if clip_to_num_tokens is not None:
                answer_in_doc = clip_text(
                    answer_in_doc,
                    num_tokens=clip_to_num_tokens,
                    model_name=model_name,
                )
            full_answer_in_doc = format_with_metadata(
                content=answer_in_doc, document=doc
            )
            return full_answer_in_doc
        else:
            return None

    def build_executor():
        if parallel and num_threads >= 1:
            executor = ThreadPoolExecutor(num_threads)
            map_fn = executor.map
        else:
            executor = contextlib.nullcontext()
            map_fn = map
        return executor, map_fn

    executor, map_fn = build_executor()

    with executor:
        selected_content = list(map_fn(prepare_for_summary, documents))

    return selected_content


def get_google_search_phrase(
    search_string: str, model_name: str = DEFAULT_FLAGSHIP_MODEL
):
    bot = Bot(name="bjork_googler", model_name=model_name)
    user_prompt = bot.format_user_prompt(search_string=search_string)
    bot.history.add_user_event(user_prompt)
    search_phrase = bot.complete(max_tokens=1024, temperature=0.1)["content"]
    # Remove the prefix, log and return
    search_phrase = search_phrase.strip("`")
    logger.debug(search_phrase)
    return search_phrase


def people_lookup(
    description: str,
    find_contact_details: bool = False,
    model_name: str = DEFAULT_FLAGSHIP_MODEL,
) -> Tuple[str, bool]:
    from althea.user import rocket_reach_search, rocket_reach_contact_lookup

    # The first step is to look up the person with
    lookup_bot = Bot(name="people_lookup_keywords", model_name=model_name)
    user_prompt = lookup_bot.format_user_prompt(description=description)
    lookup_bot.history.add_user_event(user_prompt)

    lookup_query = None
    for try_num in range(3):
        lookup_query = lookup_bot.complete(max_tokens=1024, temperature=0.1)["content"]
        logger.debug(lookup_query)
        try:
            lookup_query = parse_json(lookup_query, raise_on_fail=True)
        except JSONDecodeError as e:
            logger.warning("Failed to parse the response. Retrying.")
            lookup_bot.history.add_user_event(
                f"# System Message\n\n"
                f"The response was not a valid json block. "
                f"Error: {str(e)}. Please try again."
            )
            continue
        break

    if lookup_query is None:
        return DESCRIPTION_NOT_PARSEABLE, False

    # FIXME: lookup_query can also be a list, if the user has provided a list of queries.
    #  In that case, we should loop through the queries and the results
    if isinstance(lookup_query, (list, tuple)):
        bios = []
        for q in lookup_query:
            new_description = f"I'm looking for this person: {str(q)}"
            bio, success = people_lookup(
                description=new_description,
                find_contact_details=find_contact_details,
                model_name=model_name,
            )
            if success:
                bios.append(bio)
        if len(bios) == 0:
            return PERSON_NOT_FOUND, False
        else:
            return "\n\n---\n\n".join(bios), True
    else:
        # Use the keywords to find the person
        persons_of_interest = rocket_reach_search(**lookup_query)

    if len(persons_of_interest) == 0 and lookup_query["name"] is not None:
        # It's possible that rocket reach has this person, but under the wrong title.
        # Let's try first without the title
        persons_of_interest = rocket_reach_search(
            **dict(lookup_query, current_title=None)
        )

    if len(persons_of_interest) == 0 and lookup_query["current_employer"] is not None:
        # It's possible that rocket reach has this person, but under the wrong company.
        # Let's try first without the company
        persons_of_interest = rocket_reach_search(
            **dict(lookup_query, current_employer=None)
        )

    if (
        len(persons_of_interest) == 0
        and lookup_query["name"] is not None
        and lookup_query["current_employer"] is not None
    ):
        # It's possible that rocket reach has this person, but under the wrong title or company.
        # Let's try first without the title and company
        persons_of_interest = rocket_reach_search(
            **dict(lookup_query, current_title=None, current_employer=None)
        )

    # Check if the person is right
    if len(persons_of_interest) == 0:
        return PERSON_NOT_FOUND, False

    right_person = None
    for idx, person in enumerate(persons_of_interest):
        # Check if person matches the description
        match_check_bot = Bot(
            name="people_lookup_match_check", model_name=DEFAULT_FLAGSHIP_MODEL
        )
        user_prompt = match_check_bot.format_user_prompt(
            description=description, person=json.dumps(dict(person), indent=2)
        )
        logger.debug(user_prompt)
        match_check_bot.history.add_user_event(user_prompt)
        match_check = match_check_bot.complete(max_tokens=1024, temperature=0.1)[
            "content"
        ]
        logger.debug(match_check)
        match_check = parse_json(match_check)
        if match_check["match"]:
            right_person = person
            break
        if idx > 3:
            break
    # If the right person was not found, we return
    if right_person is None:
        return PERSON_NOT_FOUND, False

    # Find the bio of the person
    biobot = Bot(name="biobot", model_name=model_name)
    biobot.history.add_user_event(json.dumps(dict(right_person), indent=2))
    bio = biobot.complete(max_tokens=1024, temperature=0.1)["content"]
    logger.debug(bio)

    # If we don't want to find contact details, we return the bio
    if not find_contact_details:
        return bio, True

    # If we want to find contact details, we do that using the ID
    contact_details = rocket_reach_contact_lookup(
        person_id=right_person.get("id"),
        linkedin_url=right_person.get("linkedin_url"),
    )

    # Contact details not found
    if contact_details is None:
        return bio, False

    # Contact details found. Append them to the bio
    bio = f"{bio}\n\n# Contact Details\n\n"
    if linkedin_url := contact_details.get("linkedin_url", None):
        bio += f"[LinkedIn Profile]({linkedin_url})\n"
    if work_email := contact_details.get("recommended_professional_email", None):
        bio += f"Work Email: {work_email}\n"
    if personal_email := contact_details.get("recommended_personal_email", None):
        bio += f"Personal Email: {personal_email}\n"

    return bio, True
