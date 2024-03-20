import threading

from PyPDF2 import PdfReader
from io import BytesIO
import base64
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Union, Any, Tuple
import re
from urllib.parse import urlparse, parse_qs

import bs4
import numpy as np
import scipy

import backoff
import goose3
import requests

from apify_client import ApifyClient
from apify_client._errors import ApifyApiError
from gnews import GNews
from goose3 import Goose, Configuration
from common.logger import logger
from urlextract import URLExtract

from bot import embed, Bot
from retrieve import retrieve
from utils import (
    custom_memoize,
    to_normed_array,
    normalize_scores,
    clip_text,
    count_tokens_in_str,
    find_text_under_header,
    DEFAULT_FLAGSHIP_MODEL,
    DEFAULT_TOKENIZER_MODEL,
    DEFAULT_VISION_MODEL,
    DEFAULT_FAST_MODEL,
)
from utils import flatten_dict, format_string, url_to_domain, get_path, get_key


class WebScraper:
    USER_AGENTS = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.3",
        # Firefox on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:92.0) Gecko/20100101 Firefox/92.0",
        # Safari on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        # Edge on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.3 Edg/94.0.992.31",
        # Opera on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.3 OPR/64.0.3417.54",
        # Internet Explorer 11
        "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
        # Chrome on Android
        "Mozilla/5.0 (Linux; Android 10; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.181 Mobile Safari/537.3",
        # Firefox on Android
        "Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/68.0",
        # Safari on iOS
        "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
        # Chrome on iOS
        "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) CriOS/92.0.4515.159 Mobile/15E148 Safari/604.1",
    ]

    @staticmethod
    def download_and_parse_pdf(url: str) -> Optional[str]:
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Parse the PDF
            with BytesIO(response.content) as open_pdf_file:
                read_pdf = PdfReader(open_pdf_file)
                text = ""
                for page in read_pdf.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.debug(f"Failed to download and parse PDF for {url} with error {e}")
            return None

    @staticmethod
    def is_pdf(url):  # Added a default timeout duration of 10 seconds
        if url.endswith(".pdf"):
            return True
        try:
            response = requests.head(
                url,
                allow_redirects=True,
                timeout=60,
            )
            content_type = response.headers.get("Content-Type")
            return content_type == "application/pdf"
        except requests.RequestException as e:
            print(f"Error: {e}")
            return False

    @classmethod
    @custom_memoize(expire=604800, tag="goose")
    @backoff.on_exception(
        backoff.expo, goose3.network.NetworkError, max_time=30, max_tries=2
    )
    def fetch_content(cls, url) -> Optional[Dict[str, Any]]:
        result = None
        min_num_acceptable_tokens = 64  # ~50 words
        max_num_tokens_for_vision = 1536  # ~1000 words

        # if goose fails, try PDF
        if WebScraper.is_pdf(url):
            logger.debug(f"Webscraper.fetch_content failed for {url} - trying PDF")
            result = cls.download_and_parse_pdf(url)
            if result is None:
                logger.debug(f"PDF failed for {url} - trying fallback")
            else:
                result = {"url": url, "content": result, "title": None}

        if result is None:
            num_tokens = 0
        else:
            num_tokens = count_tokens_in_str(result["content"], DEFAULT_TOKENIZER_MODEL)

        if result is None or (num_tokens < min_num_acceptable_tokens):
            logger.debug("Crawling with Apify.")
            result = WebScraper.apify_crawl_url(url)

        if result is None:
            num_tokens = 0
        else:
            num_tokens = count_tokens_in_str(result["content"], DEFAULT_TOKENIZER_MODEL)

        # If the number of tokens is less than min_num_acceptable_tokens,
        # we process the screenshot.
        if result is not None and WebScraper.check_caught(result["content"]):
            if (
                result is not None
                and result.get("screenshot_url") is not None
                and num_tokens < max_num_tokens_for_vision
            ):
                logger.debug(
                    f"Processing screenshot {result['screenshot_url']} with vision."
                )
                result = WebScraper._process_with_vision(result)

        if result is None:
            logger.debug("Crawling with Goose.")
            # Try goose
            result = WebScraper._fallback_to_goose(url)

        return result

    @staticmethod
    def check_caught(page_content: str):
        bot = Bot(name="scrape_checker", model_name=DEFAULT_FAST_MODEL)
        stripped_page_content = clip_text(
            page_content, num_tokens=256, model_name=DEFAULT_FAST_MODEL
        )
        user_prompt = bot.format_user_prompt(page_content=stripped_page_content)
        bot.history.add_user_event(user_prompt)
        logger.debug(user_prompt)

        # The first step is to clarify the intention of the client and to get a new question
        truth_value = bot.complete(
            max_tokens=1024, temperature=0.1, handle_overflow=True
        )["content"]
        if truth_value == "True":
            return True
        elif truth_value == "False":
            return False
        else:
            logger.error(
                f"Truth value was {truth_value}, not true or false. Assuming true."
            )
            return True

    @staticmethod
    def _fallback_to_goose(url) -> Optional[Dict]:
        goose_config = Configuration()
        goose_config.browser_user_agent = random.choice(WebScraper.USER_AGENTS)
        goose_instance = Goose(goose_config)
        try:
            page = goose_instance.extract(url=url)
            if page.cleaned_text:
                logger.debug(f"Goose worked. content: {page.cleaned_text[:1000]}...")
                return {"url": url, "title": page.title, "content": page.cleaned_text}
        except Exception as e:
            logger.debug(f"Goose for {url} failed with {e}")
        return None

    def _fallback_to_newspaper3k(self, url):
        # FIXME: Move this back up when integrated
        from newspaper import Article

        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.text:
                logger.debug(f"Newspaper3k worked. content: {article.text[:1000]}...")
                return {"url": url, "title": article.title, "content": article.text}
            else:
                logger.debug(f"Newspaper3k for {url} failed with no text")
        except Exception as e:
            logger.debug(f"Newspaper3k for {url} failed with {e}")
        return None

    @staticmethod
    def apify_crawl_url(url: str) -> Optional[Dict]:
        client = ApifyClient(get_key("apify"))

        # Prepare the actor input
        run_input = {
            "aggressivePrune": False,
            "clickElementsCssSelector": '[aria-expanded="false"]',
            "crawlerType": "playwright:firefox",
            "debugLog": False,
            "debugMode": False,
            "dynamicContentWaitSecs": 10,
            "htmlTransformer": "readableText",
            "initialConcurrency": 0,
            "maxConcurrency": 200,
            "maxCrawlDepth": 0,
            "maxCrawlPages": 1,
            "maxResults": 9999999,
            "maxScrollHeightPixels": 5000,
            "proxyConfiguration": {"useApifyProxy": True},
            "readableTextCharThreshold": 100,
            "removeCookieWarnings": True,
            "removeElementsCssSelector": 'nav, footer, script, style, noscript, svg,\n[role="alert"],\n[role="banner"],\n[role="dialog"],\n[role="alertdialog"],\n[role="region"][aria-label*="skip" i],\n[aria-modal="true"]',
            "requestTimeoutSecs": 60,
            "saveFiles": False,
            "saveHtml": False,
            "saveMarkdown": True,
            "saveScreenshots": True,
            "startUrls": [{"url": url}],
        }

        # Sometimes, Apify returns an error that says
        # "By launching this job you will exceed the memory limit of X MB."
        # This means that too many processes are running concurrently, and if
        # we wait a bit, it will work.
        def giveup_handler(error):
            message = "By launching this job you will exceed the memory limit of"
            return message not in str(error)

        @backoff.on_exception(
            backoff.constant,
            ApifyApiError,
            giveup=giveup_handler,
            max_tries=10,
            interval=15,
        )
        def call_apify():
            return client.actor("apify/website-content-crawler").call(
                run_input=run_input
            )

        # Run the actor and wait for it to finish
        try:
            run = call_apify()
            items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            if items:
                item = items[0]
                return {
                    "url": url,
                    "title": item.get("metadata", {}).get("title"),
                    "content": item["text"],
                    "screenshot_url": item.get("screenshotUrl", None),
                }
        except ApifyApiError as e:
            logger.error(f"Apify crawler failed for {url} with {e}")
        return None

    @staticmethod
    def _process_with_vision(scrape_result: Dict[str, Any]) -> Dict[str, Any]:
        scrape_result = dict(scrape_result)
        bot = Bot(name="web_page_reader", model_name=DEFAULT_VISION_MODEL)
        user_prompt = bot.format_user_prompt(
            url=scrape_result["url"], content=scrape_result["content"]
        )
        logger.debug(user_prompt)
        bot.history.add_user_event(
            user_prompt, image_url=scrape_result["screenshot_url"]
        )
        result = bot.complete(max_tokens=1024, temperature=0.2, handle_overflow=False)
        logger.debug(result.content)
        extracted_content = find_text_under_header(result.content, "# Enriched Content")
        if extracted_content is not None:
            scrape_result["content"] = extracted_content
        return scrape_result

    @staticmethod
    def _fallback_to_soup(url: str) -> Optional[Dict]:
        def extract_text(element):
            if isinstance(element, (bs4.element.Comment, bs4.element.Doctype)) or (
                isinstance(element, bs4.element.Tag)
                and element.name in ["script", "style"]
            ):
                return ""
            if (
                isinstance(element, bs4.element.Tag)
                and element.name == "a"
                and element.get_text(strip=True)
            ):
                return f"[{element.get_text(strip=True)}]"
            if isinstance(element, bs4.element.Tag):
                return " | ".join(
                    filter(None, [extract_text(child) for child in element.children])
                )
            return element.strip()

        try:
            response = requests.get(url, verify=False)
            soup = bs4.BeautifulSoup(response.content, "html.parser")

            main_content = (
                "| "
                + " | ".join(
                    filter(None, [extract_text(child) for child in soup.children])
                )
                + " |"
            )
            main_content = re.sub("\t+", " <tab> ", main_content)
            main_content = re.sub("\n+", "\n", main_content)
            main_content = re.sub(" +", " ", main_content).strip()
            logger.debug(f"Soup for {url} worked, content: {main_content[:1000]}...")
            return dict(url=url, title=soup.title.text, content=main_content)
        except Exception as e:
            logger.debug(f"Soup for {url} failed with {e}")
        return None

    @staticmethod
    def get_linkedin_from_rocketreach(url) -> Optional[Dict]:
        if "linkedin.com" not in url:
            return None

        from user import rocket_reach_search

        results = rocket_reach_search(keyword=url)
        if len(results) == 0:
            return None

        result = dict(results[0])

        if "links" in result:
            result.pop("links")
        if "profile_pic" in result:
            result.pop("profile_pic")
        if "teaser" in result:
            result.pop("teaser")

        # Convert the structured result to a text blurb
        biobot = Bot(name="biobot", model_name="mistralai/Mixtral-8x7B-Instruct-v0.1")
        biobot.history.add_user_event(json.dumps(result, indent=2))
        result = biobot.complete(max_tokens=512, temperature=0.2, handle_overflow=False)
        return result


class URLSlugs:
    def __init__(self):
        self._slug_registry = {}
        self._url_extractor = URLExtract()
        self._lock = threading.Lock()

    @staticmethod
    def url_to_slug(url: str, prefix: Optional[str] = "sources") -> str:
        bot = Bot(name="url_slugger", model_name=DEFAULT_FAST_MODEL)
        user_prompt = bot.format_user_prompt(url=url)
        bot.history.add_user_event(user_prompt)
        slug = bot.complete(temperature=0.2, max_tokens=10).content.strip()
        if prefix is None:
            return slug
        else:
            return f"{prefix}/{slug}"

    def get_slug(self, url: str) -> str:
        with self._lock:
            if url in self._slug_registry:
                return self._slug_registry[url]
            else:
                slug = self.url_to_slug(url)
                # Check if the slug is already in use. If it is, do {slug}-{next_number}
                if slug in self._slug_registry.values():
                    i = 1
                    while f"{slug}-{i}" in self._slug_registry.values():
                        i += 1
                    slug = f"{slug}-{i}"
                self._slug_registry[url] = slug
                return slug

    def get_slugs(self, urls: List[str]) -> List[str]:
        return [self.get_slug(url) for url in urls]

    def get_url_for_slug(self, slug: str) -> Optional[str]:
        with self._lock:
            for url, s in self._slug_registry.items():
                if s == slug:
                    return url
            return None

    def convert_urls_to_slugs_in_text(self, text: str) -> str:
        urls = list(self._url_extractor.find_urls(text, only_unique=True))
        for url in urls:
            slug = self.get_slug(url)
            text = text.replace(url, slug)
        return text

    def convert_slugs_to_urls_in_text(self, text: str) -> str:
        # Assuming slugs are formatted as "sources/slug-name" in the text
        slug_pattern = re.compile(r"\bsources/[\w-]+/?\b")
        slugs = slug_pattern.findall(text)

        for slug in slugs:
            # Remove any trailing slashes
            slug = slug.rstrip("/")
            # Remove the prefix from the slug to match the format in the registry
            formatted_slug = slug
            url = self.get_url_for_slug(formatted_slug)
            if url:
                text = text.replace(slug, url)
        return text

    def reset(self) -> None:
        self._slug_registry = {}


SLUGS = URLSlugs()


@dataclass
class WebPage:
    # Main attributes
    url: str
    title: str
    # Optional attributes
    # These are the links that are found in the content of the page
    links: Optional[Dict[str, str]] = None
    # Description of the page, if available
    description: Optional[str] = None
    # Date of the page, if available
    date: Optional[str] = None
    # Private attributes
    _main_content: Optional[str] = None
    _pull_failed: bool = False
    _embeddings: Dict[Tuple[str, str], List[float]] = field(default_factory=dict)
    _summaries: Dict[str, str] = field(default_factory=dict)
    _screenshot_url: Optional[str] = None

    @classmethod
    def from_url(cls, url: str) -> "WebPage":
        article = WebScraper.fetch_content(url=url)
        if article is not None:
            return cls(
                url=url,
                title=article.get("title", ""),
                _main_content=article.get("content", ""),
                _screenshot_url=article.get("screenshot_url"),
            )
        else:
            return cls(
                url=url,
                title="",
            )

    @classmethod
    def from_google_search(cls, search_result: Dict[str, str]) -> "WebPage":
        return cls(
            url=search_result["link"],
            title=search_result["title"],
            description=search_result["description"],
            date=search_result.get("date"),
        )

    def title_and_content_as_dict(self):
        return dict(title=self.title, main_content=self._main_content)

    def get_metadata(self, include_description: bool = False) -> dict:
        metadata = dict(url=self.url, title=self.title, age_in_hours=self.age_in_hours)
        if include_description:
            metadata["description"] = self.description
        return metadata

    def pull_main_content(self) -> "WebPage":
        article = WebScraper.fetch_content(url=self.url)
        if article is None:
            self._pull_failed = True
            article = {"content": "Error: failed to load web page.", "title": None}
        self._main_content = article.get("content", "")
        self._screenshot_url = article.get("screenshot_url")
        self.title = article.get("title", "")
        return self

    @property
    def main_content(self) -> str:
        if self._main_content is None:
            self.pull_main_content()
        return self._main_content

    @property
    def screenshot_url(self) -> Optional[str]:
        return self._screenshot_url

    @property
    def metadata(self):
        return self.get_metadata(include_description=True)

    @property
    def pull_failed(self):
        return self._pull_failed

    @property
    def domain(self):
        return url_to_domain(self.url)

    def get_age(self, unit: str = "days", default: Any = None) -> int:
        if self.date is None:
            return default
        try:
            article_time = datetime.strptime(self.date, "%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            article_time = datetime.strptime(self.date, "%Y-%m-%dT%H:%M:%S.%fZ")

        current_time = datetime.utcnow()
        delta = current_time - article_time
        total_hours = delta.days * 24 + delta.seconds // 3600
        if unit == "days":
            return total_hours // 24
        elif unit == "hours":
            return total_hours
        else:
            raise ValueError(f"Unit {unit} not supported")

    @property
    def age_in_hours(self) -> int:
        return self.get_age(unit="hours")

    def get_embedding(
        self, of: str = "full_content", model_name: str = "text-embedding-ada-002"
    ) -> List[float]:
        # Return cached if possible
        if (of, model_name) in self._embeddings:
            return self._embeddings[of, model_name]

        if of == "full_content":
            lines = [
                f"Title: {self.title}",
                f"URL: {self.url}",
                f"---",
                f"{self.main_content}",
            ]
            text = "\n".join(lines)
        elif of == "description":
            text = self.description
        elif of == "title":
            text = self.title
        elif of == "metadata":
            lines = [
                f"Title: {self.title}",
                f"URL: {self.url}",
                f"Description: {self.description}",
            ]
            text = "\n".join(lines)
        else:
            raise ValueError(f"of={of} not supported")

        embedding = embed(text=text, model_name=model_name)
        self._embeddings[of, model_name] = embedding
        return embedding

    def summarize(self, model_name: str = DEFAULT_FAST_MODEL) -> str:
        from bot import Bot

        if model_name in self._summaries:
            return self._summaries[model_name]

        bot = Bot(
            name="web_page_summarize",
            model_name=model_name,
            fallback_when_out_of_context=True,
        )
        user_prompt = bot.format_user_prompt(main_content=self.main_content)
        bot.history.add_user_event(user_prompt)
        response = bot.complete(max_tokens=1024, temperature=0.1, handle_overflow=True)
        response = response["content"]

        self._summaries[model_name] = response
        return response

    def ask(
        self,
        question: str,
        model_name: str = DEFAULT_FLAGSHIP_MODEL,
        answer_format_string: Optional[str] = None,
        non_answer_format_string: Optional[str] = None,
        **retriever_kwargs,
    ) -> str:
        answer, success = retrieve(
            question=question,
            content=self.main_content,
            answer_model_name=model_name,
            keyword_model_name=model_name,
            screenshot_url=self.screenshot_url,
            full_return=True,
            **retriever_kwargs,
        )
        if success and answer_format_string is not None:
            assert "{answer}" in answer_format_string, (
                "answer_format_string must contain {answer} " "if it is not None."
            )
            answer = format_string(answer_format_string, answer=answer)
        if not success and non_answer_format_string is not None:
            answer = format_string(non_answer_format_string, answer=answer)
        return answer


@dataclass
class NewsArticle:
    title: str
    description: str
    date: str
    publisher_title: str
    publisher_url: str
    google_rss_url: Optional[str] = None
    url: Optional[str] = None

    def __post_init__(self):
        if self.url is None and self.google_rss_url is not None:
            self.url = self.convert_google_rss_url_to_url(self.google_rss_url)

    @classmethod
    def from_gnews_search_result(cls, search_result: Dict[str, Any]) -> "NewsArticle":
        return cls(
            title=search_result["title"],
            description=search_result["description"],
            date=search_result["published date"],
            publisher_title=search_result["publisher"]["title"],
            publisher_url=search_result["publisher"]["url"],
            google_rss_url=search_result["url"],
        )

    @classmethod
    def search(
        cls, term: str, num_results: int = 20, period: str = "7d"
    ) -> List["NewsArticle"]:
        # https://github.com/ranahaani/GNews for args
        google_news = GNews(max_results=num_results, period=period)
        news = google_news.get_news(term)
        return [cls.from_gnews_search_result(search_result) for search_result in news]

    @staticmethod
    def convert_google_rss_url_to_url(url: str):
        parsed_url = urlparse(url)

        # Check if 'consent.google.com' is in the netloc (domain)
        if "consent.google.com" in parsed_url.netloc:
            # If present, extract the 'continue' parameter and parse it
            query_params = parse_qs(parsed_url.query)
            continue_url = query_params.get("continue")
            if continue_url:
                continue_url = continue_url[0]
                parsed_continue_url = urlparse(continue_url)
                encoded_url = parsed_continue_url.path.split("/")[-1]
            else:
                raise ValueError(
                    f"Could not parse Google RSS URL: {url}. "
                    f"No 'continue' parameter found."
                )
        else:
            # If not present, directly extract the encoded URL from the path
            encoded_url = parsed_url.path.split("/")[-1]

        def extract_first_url_from_bytes(byte_data):
            # Convert the bytes to string using ISO-8859-1 encoding
            data_str = byte_data.decode("ISO-8859-1")

            # Use a regular expression to find the first URL in the string
            match = re.search(r'https?://[^\s<>"]+|www\.[^\s<>"]+', data_str)

            if match:
                url = match.group(0)
                # Find the position of the first non-URL character after the URL
                pos = re.search(r"[^a-zA-Z0-9:/.\-_~?&=#%]", url)
                if pos:
                    # Truncate the string at the position of the non-URL character
                    url = url[: pos.start()]
                return url
            else:
                return None

        # Make sure the len is a multiple of 4
        encoded_url += "=" * (4 - len(encoded_url) % 4)
        decoded_bytes = base64.urlsafe_b64decode(encoded_url)[4:-3]
        decoded_url = extract_first_url_from_bytes(decoded_bytes)
        return decoded_url

    @staticmethod
    def convert_date_format(date_string):
        # Parse the original date string
        date_object = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %Z")

        # Format the date to the desired format
        formatted_date = date_object.strftime("%Y-%m-%dT%H:%M:%SZ")

        return formatted_date

    def to_web_page(self) -> WebPage:
        return WebPage(
            url=self.url,
            title=self.title,
            description=self.description,
            date=self.convert_date_format(self.date),
        )


def crawl(url: str, depth: int = 1, max_num_pages: int = 50) -> List[dict]:
    client = ApifyClient(get_key("apify"))

    # Prepare the Actor input
    run_input = {
        "startUrls": [{"url": url}],
        "crawlerType": "playwright:firefox",
        "excludeUrlGlobs": [],
        "maxCrawlDepth": depth,
        "maxCrawlPages": max_num_pages,
        "initialConcurrency": 0,
        "maxConcurrency": 200,
        "initialCookies": [],
        "proxyConfiguration": {"useApifyProxy": True},
        "dynamicContentWaitSecs": 10,
        "maxScrollHeightPixels": 5000,
        "removeElementsCssSelector": """nav, footer, script, style, noscript, svg,
[role=\"alert\"],
[role=\"banner\"],
[role=\"dialog\"],
[role=\"alertdialog\"],
[role=\"region\"][aria-label*=\"skip\" i],
[aria-modal=\"true\"]""",
        "removeCookieWarnings": True,
        "clickElementsCssSelector": '[aria-expanded="false"]',
        "htmlTransformer": "readableText",
        "readableTextCharThreshold": 100,
        "aggressivePrune": False,
        "debugMode": False,
        "debugLog": False,
        "saveHtml": True,
        "saveMarkdown": True,
        "saveFiles": False,
        "saveScreenshots": False,
        "maxResults": 9999999,
    }

    # Run the Actor and wait for it to finish
    run = client.actor("aYG0l9s7dbB7j3gbS").call(run_input=run_input)

    items = [
        flatten_dict(l, sep="_")
        for l in client.dataset(run["defaultDatasetId"]).iterate_items()
    ]
    return items


def shame_crawl(
    url: str,
    depth: int,
    max_num_pages: int,
    get_or_create: bool = True,
    root_dir: Optional[str] = None,
) -> List[WebPage]:
    json_path = get_path(
        f"data/crawls/{url_to_domain(url)}.json", root_directory=root_dir
    )
    if os.path.exists(json_path) and get_or_create:
        with open(json_path, "r") as f:
            result = json.load(f)
    else:
        logger.debug(f"Shame Crawling {url}.")
        result = crawl(url=url, depth=depth, max_num_pages=max_num_pages)
        logger.debug(f"Shame Crawling {url} finished.")
        with open(json_path, "w") as f:
            json.dump(result, f)
    # Compile docs
    docs = [
        WebPage(url=r["url"], title=r["metadata_title"], _main_content=r["markdown"])
        for r in result
    ]
    return docs


def sort_web_pages_by_relevance_and_recency(
    web_pages: List[WebPage],
    search_terms: List[str],
    embed_full_text: bool = False,
    return_scores: bool = False,
    relevance_weight: float = 0.5,
    recency_weight: float = 0.5,
    num_quantiles: int = 5,
    relevance_temperature: float = 0.1,
) -> Union[List[WebPage], Tuple[List[WebPage], List[float]]]:
    # Embed all search terms
    search_term_embeddings = [embed(term) for term in search_terms]

    # Embed all web pages
    web_page_embeddings = [
        web_page.get_embedding(of=("full_content" if embed_full_text else "metadata"))
        for web_page in web_pages
    ]

    # Compute the cosine similarity between each web page and each search term
    # similarity_matrix.shape = (num_web_pages, num_search_terms)
    similarity_matrix = np.dot(
        to_normed_array(web_page_embeddings), to_normed_array(search_term_embeddings).T
    )

    # Compute the relevance score for each web page
    # softmax_matrix.shape = (num_web_pages, num_search_terms)
    softmax_matrix = scipy.special.softmax(
        similarity_matrix / relevance_temperature, axis=1
    )
    relevance_scores = np.sum(softmax_matrix * similarity_matrix, axis=1)
    # relevance_scores = np.mean(similarity_matrix, axis=1)
    relevance_scores = normalize_scores(relevance_scores)

    # Compute the recency score for each web page
    ages = np.array([wp.age_in_hours for wp in web_pages])
    quantiles = np.percentile(ages, np.linspace(0, 100, num_quantiles + 1)[1:-1])
    recency_scores = np.zeros_like(ages)
    for i in range(num_quantiles):
        if i == 0:
            recency_scores[ages <= quantiles[i]] = 1 - i / num_quantiles
        elif i == num_quantiles - 1:
            recency_scores[ages > quantiles[i - 1]] = 1 - i / num_quantiles
        else:
            recency_scores[(ages > quantiles[i - 1]) & (ages <= quantiles[i])] = (
                1 - i / num_quantiles
            )

    # Combine relevance and recency scores
    combined_scores = (
        relevance_weight * relevance_scores + recency_weight * recency_scores
    )

    # Sort the web pages by combined scores
    sorted_indices = np.argsort(-combined_scores)
    sorted_web_pages = [web_pages[i] for i in sorted_indices]
    sorted_scores = combined_scores[sorted_indices]

    if return_scores:
        return sorted_web_pages, sorted_scores.tolist()
    else:
        return sorted_web_pages


def deduplicate_web_pages(
    web_pages: List[WebPage],
    by: str = "full_content",
    threshold: Optional[float] = 0.9,
) -> List[WebPage]:
    # Articles with the same URL are considered duplicates
    unique_urls = set()
    unique_web_pages = []
    for web_page in web_pages:
        if web_page.url not in unique_urls:
            unique_urls.add(web_page.url)
            unique_web_pages.append(web_page)

    if threshold is None:
        return unique_web_pages

    # Embed all web pages
    web_page_embeddings = [
        web_page.get_embedding(of=by) for web_page in unique_web_pages
    ]

    # Compute the cosine similarity between each web page and each other web page
    # similarity_matrix.shape = (num_web_pages, num_web_pages)
    similarity_matrix = np.dot(
        to_normed_array(web_page_embeddings), to_normed_array(web_page_embeddings).T
    )

    # Deduplicate based on similarity
    num_web_pages = len(unique_web_pages)
    is_duplicate = np.zeros(num_web_pages, dtype=bool)
    for i in range(num_web_pages):
        for j in range(i + 1, num_web_pages):
            if similarity_matrix[i, j] > threshold:
                is_duplicate[j] = True

    # Collect indices of unique web pages
    unique_indices = np.where(~is_duplicate)[0]

    # Select unique web pages
    deduplicated_web_pages = [unique_web_pages[i] for i in unique_indices]

    return deduplicated_web_pages
