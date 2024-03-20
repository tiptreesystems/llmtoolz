import ast
import sys
import io
import traceback
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Callable

import numpy
import pandas as pd

from utils import count_tokens_in_str, DEFAULT_TOKENIZER_MODEL


def indent(message: str, level: int = 4) -> str:
    return "\n".join([f"{' ' * level}{line}" for line in message.split("\n")])


class LevelOfDetail(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class EvaluationResult:
    stdout: Optional[str]
    stderr: Optional[str]
    state_change_summary: Optional[str]
    current_state_summary: Optional[str]

    def __post_init__(self):
        if self.stdout == "":
            self.stdout = None
        else:
            self.stdout = self.stdout.strip("\n")
        if self.stderr == "":
            self.stderr = None
        else:
            self.stderr = self.stderr.strip("\n")
        if self.state_change_summary == "":
            self.state_change_summary = None
        else:
            self.state_change_summary = self.state_change_summary.strip("\n")
        if self.current_state_summary == "":
            self.current_state_summary = None
        else:
            self.current_state_summary = self.current_state_summary.strip("\n")

    def express_state_change(self) -> str:
        sub_strings = []
        if self.stdout is not None:
            sub_strings.append("The following was printed to stdout:\n")
            sub_strings.append(indent(self.stdout, level=2))
            sub_strings.append("\n")
        if self.stderr is not None:
            sub_strings.append("The following error was raised:\n")
            sub_strings.append(indent(self.stderr, level=2))
            sub_strings.append("\n")
        if self.state_change_summary is not None:
            sub_strings.append(self.state_change_summary)
            sub_strings.append("\n")
        return "".join(sub_strings).strip("\n")

    def express_state(self) -> str:
        sub_strings = []
        if self.current_state_summary is not None:
            sub_strings.append(self.current_state_summary)
            sub_strings.append("\n")
        return "".join(sub_strings).strip("\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "state_change_summary": self.state_change_summary,
            "current_state_summary": self.current_state_summary,
        }

    def __repr__(self) -> str:
        return f"EvaluationResult({self.to_dict()})"


class REPL:
    def __init__(
        self,
        existing_variables: Optional[Dict[str, Any]] = None,
        available_functions: Optional[Dict[str, Callable]] = None,
        max_num_tokens_per_print: int = 7000,
        str_functions: Optional[dict] = None,
    ):
        self.module_aliases = {
            "np": numpy,
            "pd": pd,
            "numpy": numpy,
            "pandas": pd,
        }
        if available_functions is None:
            available_functions = dict()
        self.available_functions = available_functions
        if str_functions is None:
            str_functions = dict()
        self.str_functions = str_functions
        self.max_num_tokens_per_print = max_num_tokens_per_print
        # Privates
        self._state = {}
        self.latest_added_variable_names = []
        if existing_variables is not None:
            self._state.update(existing_variables)

    def add_to_state(self, variables: Dict[str, Any]):
        # latest_added_variable_name
        self.latest_added_variable_names = list(variables.keys())
        self._state.update(variables)
        return self

    def summarize_variable(
        self, variable: Any, level_of_detail: LevelOfDetail = LevelOfDetail.HIGH
    ) -> str:
        if isinstance(variable, pd.DataFrame):
            return Summarizers.summarize_data_frame(variable, level_of_detail)

        if isinstance(variable, pd.Series):
            return Summarizers.summarize_series(variable)

        if isinstance(variable, numpy.ndarray):
            return Summarizers.summarize_numpy_array(variable)

        return Summarizers.summarize_any(variable)

    def summarize_current_state(self, level_of_detail: str = "high") -> str:
        if len(self._state) == 0:
            return "No variables defined yet."
        desc = "The following variables are currently defined:"
        for key, value in self._state.items():
            # Ensure value is not a module
            if isinstance(value, type(numpy)):
                continue
            # Ensure value is not a function
            if callable(value):
                continue
            # Everything else can be summarized
            desc += "\n"
            desc += indent(
                f"{key}: {self.summarize_variable(value, LevelOfDetail.MEDIUM)}", 2
            )
        desc = desc.rstrip("\n")
        return desc

    def summarize_state_change(
        self, previous_state: Dict[str, Any], new_state: Dict[str, Any]
    ) -> str:
        new_variables = set(new_state.keys()) - set(previous_state.keys())
        self.latest_added_variable_names = list(new_variables)
        deleted_variables = set(previous_state.keys()) - set(new_state.keys())
        # Ignore existing variables for now
        desc = ""
        if len(new_variables) > 0:
            desc += "The following variables were defined:"
            for key in new_variables:
                desc += "\n"
                desc += indent(
                    f"{key}: {self.summarize_variable(new_state[key], level_of_detail='low')}",
                    2,
                )
        if len(deleted_variables) > 0:
            desc += "The following variables were deleted:\n"
            for key in deleted_variables:
                desc += "\n"
                desc += indent(
                    f"{key}: {self.summarize_variable(previous_state[key], level_of_detail='low')}",
                    2,
                )
        desc = desc.rstrip("\n")
        return desc

    def to_string(self, obj: Any) -> str:
        for type_, func in self.str_functions.items():
            if isinstance(obj, type_):
                return func(obj)
        return str(obj)

    def evaluate(self, code_snippet: str) -> Dict[str, Any]:
        """
        Evaluates a snippet of code in the REPL.

        Args:
            code_snippet: The code snippet to evaluate.

        """
        local_variables = dict(
            **self.module_aliases, **self.available_functions, **self._state
        )
        old_stdout = sys.stdout
        sys.stdout = exec_stdout = io.StringIO()

        try:
            # Parse the code snippet into an AST
            parsed_code = ast.parse(code_snippet)

            # Process each node in the parsed AST
            for node in parsed_code.body:
                if isinstance(node, ast.Expr):  # An expression
                    # Evaluate the expression and print the result
                    result = eval(
                        compile(
                            ast.Expression(node.value), filename="<ast>", mode="eval"
                        ),
                        local_variables,
                    )
                    result_str = self.to_string(result)
                    # If it's too many tokens (> 10k), say that it's too long
                    if (
                        count_tokens_in_str(result_str, DEFAULT_TOKENIZER_MODEL)
                        > self.max_num_tokens_per_print
                    ):
                        result_str = "[Result is too long to display.]"

                    print(result_str)
                else:  # A statement
                    exec(
                        compile(ast.Module([node], []), filename="<ast>", mode="exec"),
                        local_variables,
                    )

            stdout = exec_stdout.getvalue()
            stderr = ""
        except Exception as e:
            stdout = ""
            traceback_str = "".join(
                traceback.format_exception(None, e, e.__traceback__)
            )
            traceback_lines = traceback_str.split("\n")
            if len(traceback_lines) > 10:
                num_truncated_lines = len(traceback_lines) - 10
                traceback_str = "\n".join(
                    [f"[{num_truncated_lines} lines truncated]"] + traceback_lines[-10:]
                )
            stderr = f"Error: {str(e)}\nTraceback: {traceback_str}"
        finally:
            sys.stdout = old_stdout

        # Remove the module aliases from the local variables
        for alias in self.module_aliases.keys():
            local_variables.pop(alias)
        for alias in self.available_functions.keys():
            local_variables.pop(alias)
        # Summarize the state change
        state_change_summary = self.summarize_state_change(self._state, local_variables)
        # Update the state
        self._state = local_variables
        # Summarize the current state
        current_state_summary = self.summarize_current_state(level_of_detail="low")
        # Return the output string
        return EvaluationResult(
            stdout=stdout,
            stderr=stderr,
            state_change_summary=state_change_summary,
            current_state_summary=current_state_summary,
        ).to_dict()


class Summarizers:
    @staticmethod
    def summarize_any(obj: Any, detail: int = 0, indent: int = 0) -> str:
        if obj is None:
            return "None"
        return f"Object of type {type(obj).__name__}"

    @staticmethod
    def summarize_data_frame(
        df: pd.DataFrame, level_of_detail: LevelOfDetail = LevelOfDetail.HIGH
    ) -> str:
        # Start with the basic structure of the DataFrame
        desc_str = f"DataFrame with {df.shape[0]} rows and {df.shape[1]} columns.\n"
        if level_of_detail == LevelOfDetail.LOW:
            return desc_str
        elif level_of_detail == LevelOfDetail.MEDIUM:
            column_names = [str(col) for col in df.columns][:20]
            num_columns_left = df.shape[1] - len(column_names)
            desc_str += indent(
                f"Columns: {', '.join(column_names)}, "
                f"... ({num_columns_left} more columns hidden).\n",
                2,
            )
            return desc_str
        elif level_of_detail == LevelOfDetail.HIGH:
            # Show all columns
            column_names = [str(col) for col in df.columns]
            desc_str += indent(f"Columns: {', '.join(column_names)}.\n", 2)
            return desc_str
        return desc_str

    @staticmethod
    def summarize_series(series: pd.Series) -> str:
        series_type = series.dtype
        unique_values = series.nunique()
        if series.value_counts().size > 0:
            most_common_value = series.value_counts().idxmax()
            most_common_count = series.value_counts().max()
        else:
            most_common_value = None
            most_common_count = 0

        # Basic Series structure
        desc_str = f"Series of length {len(series)} with type {series_type}.\n"

        # Detailed description
        if series_type == "object":
            desc_str += f"Categorical with {unique_values} unique values. "
            desc_str += f'Most common value: "{most_common_value}" ({most_common_count} occurrences).'
        elif series_type in ["int64", "float64"]:
            desc_str += f"Numeric. "
            desc_str += f"Mean: {series.mean():.2f}, Median: {series.median():.2f}, Std: {series.std():.2f}."
        else:
            desc_str += f"Type: {series_type}."

        return desc_str

    @staticmethod
    def summarize_numpy_array(obj: numpy.ndarray, detail: int = 0) -> str:
        return f"Numpy array of shape {obj.shape} and dtype {obj.dtype}"


# Printers
class Printers:
    @staticmethod
    def dataframe_as_csv_printer(
        df: pd.DataFrame, max_num_tokens_before_truncation: Optional[int] = None
    ) -> str:
        result_str = str(df.to_csv(sep="\t", index=False))
        if max_num_tokens_before_truncation is not None and (
            count_tokens_in_str(result_str, DEFAULT_TOKENIZER_MODEL)
            > max_num_tokens_before_truncation
        ):
            # Result is too long to display, so we'll need to truncate
            result_str = (
                f"Dataframe with {df.shape[0]} rows and {df.shape[1]} columns. "
                f"Result is too long to display."
            )
        return result_str
