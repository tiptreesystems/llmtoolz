import atexit
import csv
import inspect
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from loguru._logger import Logger, Core


def route_to_user_if_needed(fn):
    def wrapper(self, *args, **kwargs):
        if (
            kwargs.pop("user", False)
            and getattr(self, "_bound_user_id", None) is not None
        ):
            try:
                self.user(args[0])
            except Exception as e:
                tb_str = "".join(traceback.format_exception(None, e, e.__traceback__))
                print(f"Failed to log user message: {e}\nTraceback: {tb_str}")
        return fn(self, *args, **kwargs)

    return wrapper


class CommonLogger(Logger):
    debug = route_to_user_if_needed(Logger.debug)
    warning = route_to_user_if_needed(Logger.warning)
    exception = route_to_user_if_needed(Logger.exception)
    error = route_to_user_if_needed(Logger.error)
    info = route_to_user_if_needed(Logger.info)

    def user(self, message: str):
        bound_user_id = getattr(self, "_bound_user_id", None)
        if bound_user_id is None:
            return

        try:
            self._user(message, bound_user_id)
        except Exception as e:
            self.exception(f"Failed to log user message: {e}")

    def _user(self, message: str, bound_user_id: str):
        from common.utils import get_path

        data_dir = get_path(f"data/text_rain/{bound_user_id}/")
        timestamp = datetime.utcnow().isoformat() + "Z"  # ISO 8601 format
        current_frame = inspect.currentframe()

        # Loop over outer frames until we find a frame in a different file
        while True:
            caller_frame = inspect.getouterframes(current_frame, 2)
            filename = caller_frame[1][1].split("/")[
                -1
            ]  # get the filename from the frame
            if filename != "logger.py":
                break
            current_frame = caller_frame[1][0]  # Move up one frame in the stack

        function_name = caller_frame[1][3]  # get the function name from the frame
        line_number = caller_frame[1][2]  # get the line number from the frame

        log_dict = {
            "timestamp": timestamp,
            "filename": filename,
            "function_name": function_name,
            "line_number": line_number,
            "message": message,
        }
        with open(data_dir + "user.log", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_dict.keys())
            writer.writerow(log_dict)

    @contextmanager
    def using_bound_user_id(self, user_id: str):
        self.reset()
        self.bind_user_id(user_id)
        yield
        self.reset()

    def bind_user_id(self, user_id: str):
        setattr(self, "_bound_user_id", user_id)

    def reset(self):
        if hasattr(self, "_bound_user_id"):
            delattr(self, "_bound_user_id")


logger = CommonLogger(
    core=Core(),
    exception=None,
    depth=1,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)

logger.add(sys.stderr)
atexit.register(logger.remove)
