from typing import TYPE_CHECKING
from copy import deepcopy

from utils import multiline_input

if TYPE_CHECKING:
    from bot import Bot


class BotError(Exception):
    pass


class OutOfTokenCapError(BotError):
    pass


def playground_ui(bot: "Bot"):
    # Get a copy of the bot and make it have no side effects
    bot = deepcopy(bot)
    bot.auto_persist = False

    # Print the history
    for event in bot.history.events:
        print(str(event))
        print("-" * 80)

    while True:
        # Get the user input
        user_input = multiline_input().strip()
        if user_input != "":
            # Add the user input to the history
            bot.history.add_user_event(user_input)
        elif user_input == "!q":
            break
        print("-" * 80)
        bot.complete(max_tokens=1024, temperature=0.2)
        print(str(bot.history.get_most_recent_event()))
        print("-" * 80)
