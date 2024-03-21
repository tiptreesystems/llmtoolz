from common.utils import get_path
from llmtoolz import lorimer

#     question: str,
#     *,
#     context: Optional[str] = None,
#     max_document_depth: int = 2,
#     parallel: bool = True,
#     num_threads: int = 3,
#     max_iterations: int = 5,
#     fake_it: bool = False,
#     log_path: Optional[str] = None,
#     # Kwargs for arnar
#     num_bjork_calls_in_arnar: int = 3,
#     arnar_kwargs: Optional[dict] = None,
#     # Kwargs for bjork
#     parallel_bjork: bool = False,
#     num_threads_for_bjork: int = 3,
#     max_queries_in_bjork: int = 3,
#     num_pages_to_select_in_bjork: int = 3,
#     bjork_kwargs: Optional[dict] = None,

def main():
    section = lorimer(question="Write a report that compares and contrasts ViT and ResNet")
    section.persist(get_path("data/lorimer_report.md"))
    print(section.render())


if __name__ == "__main__":
    main()