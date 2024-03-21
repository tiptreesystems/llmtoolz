import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Union, Optional, Dict

import numpy as np
from apify_client import ApifyClient

from .bot import batch_embed, embed
from .scrape import WebPage, NewsArticle
from .utils import custom_memoize as memoize
from common.utils import get_key


@dataclass
class GoogleSearch:
    query: str
    results: List[WebPage]

    @staticmethod
    @memoize(expire=3600, tag="serp")
    def search(
        search_query: Union[str, List[str]],
        num_results: int = 5,
        as_json_string: bool = True,
        json_string_indent: Optional[int] = None,
    ) -> Union[str, List[Dict[str, str]]]:
        client = ApifyClient(get_key("apify"))

        if isinstance(search_query, str):
            input_was_str = True
            search_query = [search_query]
        else:
            input_was_str = False

        # Prepare the actor input
        run_input = {
            "queries": "\n".join(search_query),
            "maxPagesPerQuery": 1,
            "resultsPerPage": num_results,
            "countryCode": "",
            "customDataFunction": """async ({ input, $, request, response, html }) => {
          return {
            pageTitle: $('title').text(),
          };
        };""",
        }

        # Run the actor and wait for it to finish
        run = client.actor("apify/google-search-scraper").call(run_input=run_input)

        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        serp_results = []
        for item in items:
            serp_result = item["organicResults"]
            # Keep only the title, link and description
            serp_result = [
                dict(
                    title=result["title"],
                    link=result["url"],
                    description=result["description"],
                    date=result.get("date", None),
                )
                for result in serp_result
                if result.get("url") is not None
            ]
            serp_results.append(serp_result)

        if input_was_str and len(serp_results) == 1:
            serp_results = serp_results[0]

        if as_json_string:
            serp_results = json.dumps(serp_results, indent=json_string_indent)
        return serp_results

    @classmethod
    def from_query(
        cls, query: str, num_results: int = 5, exact_search: bool = False
    ) -> "GoogleSearch":
        if exact_search:
            query = f'"{query}"'
        results = cls.search(query, num_results=num_results, as_json_string=False)
        # Filter results without description
        results = [
            WebPage.from_google_search(result)
            for result in results
            if result.get("description") not in [None, ""]
        ]
        return cls(query=query, results=results)

    @classmethod
    def from_queries(
        cls, queries: List[str], num_results: int
    ) -> Dict[str, "GoogleSearch"]:
        all_results = cls.search(queries, num_results=num_results, as_json_string=False)
        return_value = {}
        for query, results in zip(queries, all_results):
            # Filter results without description
            results = [
                WebPage.from_google_search(result)
                for result in results
                if result.get("description") not in [None, ""]
            ]
            return_value[query] = cls(query=query, results=results)
        return return_value

    def to_dict(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        return dict(query=self.query, results=self.results)

    def rerank(
        self, weight: float = 1.0, query: Optional[str] = None
    ) -> "GoogleSearch":
        if query is None:
            query = self.query
        self.results = rerank_search_results(query, self.results, weight=weight)
        return self

    def keep_top_results(self, num_results: int) -> "GoogleSearch":
        self.results = self.results[:num_results]
        return self


@dataclass
class NewsSearch:
    query: str
    results: List[WebPage]

    def __post_init__(self):
        self.results = [result for result in self.results if result.date is not None]
        self.results = sorted(self.results, key=lambda x: x.date, reverse=True)

    @classmethod
    def from_query(
        cls,
        query: str,
        num_results: int = 5,
        period: str = "7d",
        min_publish_date: Optional[float] = None,
    ) -> "NewsSearch":
        results = NewsArticle.search(term=query, num_results=num_results, period=period)
        web_pages = [result.to_web_page() for result in results]
        results = []
        for web_page in web_pages:
            web_page_date = datetime.strptime(web_page.date, "%Y-%m-%dT%H:%M:%SZ")

            if min_publish_date is not None:
                if web_page_date < datetime.fromtimestamp(min_publish_date):
                    continue
            else:
                results.append(web_page)
        return cls(query=query, results=results)

    @classmethod
    def from_queries(
        cls,
        queries: List[str],
        num_results: int = 5,
        return_results_as_flat_list: bool = False,
        period: str = "7d",
        min_publish_date: Optional[float] = None,
    ) -> Union[Dict[str, "NewsSearch"], List[WebPage]]:
        all_results = {}
        for query in queries:
            all_results[query] = cls.from_query(
                query,
                num_results=num_results,
                period=period,
                min_publish_date=min_publish_date,
            )
        if return_results_as_flat_list:
            all_results = [
                result
                for query_results in all_results.values()
                for result in query_results.results
            ]
        return all_results


def rerank_search_results(
    query: str,
    search_results: List[WebPage],
    embedding_model: str = "text-embedding-3-large",
    weight: float = 1.0,
) -> List[WebPage]:
    result_texts = [
        f"Title: {result.title}\n\nURL: {result.url}\n\nDescription: {result.description}"
        for result in search_results
    ]
    embeddings = np.array(batch_embed(texts=result_texts, model_name=embedding_model))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_embedding = np.array(embed(text=query, model_name=embedding_model))
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    scores = np.dot(embeddings, query_embedding.T).flatten()

    # Calculate quantiles for embedding scores
    score_quantiles = (np.argsort(np.argsort(scores)) + 1) / len(scores)

    # Calculate quantiles for initial ordering
    initial_order_quantiles = 1.0 - (
        (np.arange(len(search_results)) + 1) / len(search_results)
    )

    # Calculate weighted average of quantiles
    initial_order_weight = 1.0 - weight

    weighted_quantiles = (
        initial_order_weight * initial_order_quantiles
        + (1 - initial_order_weight) * score_quantiles
    )

    # Rerank based on weighted quantiles
    reranking_indices = np.argsort(weighted_quantiles)[::-1]
    reranked_results = [search_results[i] for i in reranking_indices]

    return reranked_results
