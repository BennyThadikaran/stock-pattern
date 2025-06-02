import json
import os
import re
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

"""
This script help you configure the user.json. Just answer the questions asked
with the choices provided.

You can use Tab key to autocomplete filepath.

File separator is a forward slash `/` for all OS platforms.

All filepaths are relative to the users home folder.
"""

try:
    import questionary
except ModuleNotFoundError:
    exit("Missing package questionary. Run `pip install questionary`")


def validate_ohlc_file(folder: Path) -> Union[bool, str]:
    """
    Check for invalid date formats, invalid columns headings

    Returns False on errors
    """
    i = 0

    expected_columns = (b"Open", b"High", b"Low", b"Close", b"Volume")
    missing_cols = "Files must have Open, High, Low, Close and Volume columns"
    invalid_date_format = "Date format is not valid."

    for file in folder.iterdir():
        # validate only three files
        if i == 3:
            break

        if not file.is_file() or os.path.getsize(file) == 0:
            continue

        with file.open("rb") as f:
            # Check the column names are correct
            columns = f.readline().strip().split(b",")

            for col in expected_columns:
                if col in columns:
                    continue

                questionary.print(
                    f"WARNING: {file.name} - {missing_cols}",
                    style=warning_color,
                )
                return False

            second_line = f.readline()

            start_date = pd.to_datetime(
                second_line[: second_line.find(b",")].decode(), errors="coerce"
            )

            if pd.isna(start_date):
                questionary.print(
                    f"WARNING: {file.name} - {invalid_date_format}",
                    style=warning_color,
                )
                return False

        i += 1

    return True


def validate_timestring(txt) -> bool:
    if ":" not in txt:
        return False

    hour, min = txt.split(":")

    return 0 <= int(hour) < 24 and 0 <= int(min) < 60


def validate_watchlist_file(file_path: Path) -> bool:
    regex = re.compile(rb"[,;\t/]")
    err_str = "Comma, semicolon and tab chars not allowed."
    count = 0

    with file_path.open("rb") as f:
        while True:
            line = f.readline().strip()

            if not line:
                break

            if re.findall(regex, line):
                questionary.print(
                    f"Line no: {count}: {line.decode()}\n{err_str}",
                    style=warning_color,
                )
                return False

    return True


def ask_default_source(user: Path) -> str:
    return questionary.path(
        "Please provide the folder path to the OHLC data. Press Tab to autocomplete",
        only_directories=True,
        get_paths=lambda: [str(user)],
        validate=lambda fpath: Path(f"{user}/{fpath}").is_dir(),
    ).ask()


def ask_loader_name() -> str:
    loader_choice_list = ("Daily or Higher timeframe", "Intraday")

    loader_choice = questionary.select(
        "What is the timeframe of your OHLC data?",
        choices=loader_choice_list,
    ).ask()

    if loader_choice == loader_choice_list[0]:
        return "EODFileLoader"

    return "IEODFileLoader"


def ask_default_timeframe(loader: str) -> str:
    if loader == "EODFileLoader":
        tfs = ("Daily", "Weekly", "Monthly")
    else:
        tfs = ("1", "5", "10", "15", "25", "30", "60", "75", "125", "2h", "4h")

    # DEFAULT OHLC TIMEFRAME
    tf = questionary.select(
        "Select the timeframe of your OHLC Data",
        choices=tfs,
    ).ask()

    return tf.lower()


def ask_market_hours():
    response = questionary.select(
        "Is the market operating 24/7? For example, cryptocurrency or forex markets.",
        choices=("Yes", "No"),
    ).ask()

    if response == "Yes":
        return True


def ask_watchlist(user: Path) -> Path:
    watchlist_path = questionary.path(
        """Please provide the filepath to the watchlist file.
    A text or CSV file with symbol names (one on each line).
    """,
        get_paths=lambda: [str(user)],
        validate=lambda fpath: (user / fpath).is_file(),
    ).ask()

    return user / watchlist_path


def main() -> Tuple[Path, dict]:
    """
    Returns the config file path and dictionary configuration
    """
    DIR = Path(__file__).parent

    config_file = DIR / "user.json"

    config = {}

    user = Path("~").expanduser()

    questionary.print(
        "# Press TAB key to autocomplete filepath or folders\n",
        style=notice_color,
    )

    questionary.print(f"# Filepaths are relative to: {user}\n", style=notice_color)

    # DEFAULT OR CUSTOM CONFIG
    if config_file.exists():
        config_choice_list = [
            "Edit user.json",
            "Edit custom config file",
            "Create custom config",
        ]

        config_choice = questionary.select(
            "What do you wish to do?",
            choices=config_choice_list,
        ).ask()

        # EDIT user.json
        if (
            config_choice == config_choice_list[0]
            or config_choice == config_choice_list[1]
        ):
            if config_choice == config_choice_list[1]:
                while True:
                    fname = questionary.text(
                        "Enter the name of the configuration file you want to edit.",
                        instruction=".json will be added to the name",
                    ).ask()

                    config_file = DIR / f"{fname}.json"

                    if not config_file.exists():
                        questionary.print(f"{config_file} not found")
                        continue
                    break

            # load user.json
            config = json.loads(config_file.read_bytes())

            field_list = (
                "Done. No more edits",
                "Change source folder - DATA_PATH",
                "Change default watchlist - SYM_LIST",
                "Change Loader class - LOADER",
                "Change default timeframe - DEFAULT_TF",
            )

            # Keep asking till user has completed edits to user.json
            while True:
                field_choice = questionary.select(
                    "Select a field to edit? Ctrl-C if you're done.",
                    choices=field_list,
                ).ask()

                if field_choice == field_list[0]:
                    # User done! break and return
                    break
                elif field_choice == field_list[1]:
                    # EDIT DATA_PATH
                    data_path = ask_default_source(user)

                    config["DATA_PATH"] = str(user / data_path)

                elif field_choice == field_list[2]:
                    # EDIT SYM_LIST
                    watchlist_path = ask_watchlist(user)

                    config["SYM_LIST"] = str(watchlist_path)
                else:
                    # EDIT LOADER and DEFAULT_TF
                    loader_choice = ask_loader_name()

                    config["LOADER"] = loader_choice

                    config["DEFAULT_TF"] = ask_default_timeframe(loader_choice)

            return config_file, config

        else:
            fname = questionary.text(
                "Please provide a name for the custom configuration file.",
                instruction=".json will be added to the name",
            ).ask()

            config_file = DIR / f"{fname}.json"

    # OHLC DATA FOLDER
    data_path = ask_default_source(user)

    print("Validating OHLC files")

    csv_check_result = validate_ohlc_file(user / data_path)

    if not csv_check_result:
        exit("Please correct the issues with the OHLC file and try again.")

    questionary.print("✓ Passed validation", style=success_color)

    config["DATA_PATH"] = str(user / data_path)

    # OHLC LOADER
    loader_choice = ask_loader_name()
    config["LOADER"] = loader_choice

    config["DEFAULT_TF"] = ask_default_timeframe(loader_choice)

    if config["LOADER"] == "IEODFileLoader":
        is_24_7 = questionary.select(
            "Is your market operating 24/7? For example Crypto or Forex market",
            choices=("Yes", "No"),
        ).ask()

        if is_24_7 == "Yes":
            config["24_7"] = True
        else:
            while True:
                exchange_start_time = questionary.text(
                    "Please provide the exchange start time in HH:MM format (e.g., 09:30).",
                ).ask()

                if validate_timestring(exchange_start_time):
                    break

            config["EXCHANGE_START_TIME"] = exchange_start_time

    # DEFAULT WATCHLIST FILE
    needs_watchlist = questionary.confirm("Add a default watchlist to scan?").ask()

    if needs_watchlist:
        watchlist_path = ask_watchlist(user)

        config["SYM_LIST"] = str(watchlist_path)

        print("Validating Watchlist file")

        if not validate_watchlist_file(watchlist_path):
            exit("Please correct issues in your watchlist file")

        questionary.print("✓ Passed validation", style=success_color)

    return config_file, config


if __name__ == "__main__":
    warning_color = "fg:red"
    success_color = "fg:green"
    notice_color = "fg:yellow"

    config_file, config = main()

    config_file.write_text(json.dumps(config, indent=2))
    print(f"Config saved: {config_file}")
