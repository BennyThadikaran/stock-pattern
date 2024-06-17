from datetime import datetime
from pathlib import Path
from typing import Tuple, BinaryIO
import pandas as pd
import json
import os
import re

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


def get_last_date(f: BinaryIO) -> datetime:
    # source: https://stackoverflow.com/a/68413780
    try:
        # seek 2 bytes to the last line ending ( \n )
        f.seek(-2, os.SEEK_END)

        # seek backwards 2 bytes till the next line ending
        while f.read(1) != b"\n":
            f.seek(-2, os.SEEK_CUR)

    except OSError:
        # catch OSError in case of a one line file
        f.seek(0)

    # we have the last line
    last_line = f.readline()
    date = last_line[: last_line.find(b",")].decode()

    return pd.to_datetime(date, errors="coerce")


def validate_ohlc_file(folder: Path) -> bool:
    """
    Check for empty files, missing columns, wrong date order and
    invalid date formats
    """
    i = 0

    is_valid = True
    expected_columns = (b"Date", b"Open", b"High", b"Low", b"Close", b"Volume")

    missing_cols = "Files must have Open, High, Low, Close and Volume columns"

    invalid_dates = "OHLC dates must be in ascending order (Oldest to Newest)"

    invalid_date_format = "OHLC date format is not valid."

    for file in folder.iterdir():
        # validate only three files
        if i == 3:
            break

        if not file.is_file() or os.path.getsize(file) == 0:
            continue

        with file.open("rb") as f:
            # Check the column names are correct
            columns = f.readline().strip(b"\n").split(b",")

            for col in expected_columns:
                if col in columns:
                    continue

                questionary.print(
                    f"WARNING: {file.name} - {missing_cols}",
                    style=warning_color,
                )
                is_valid = False

            second_line = f.readline()

            start_date = pd.to_datetime(
                second_line[: second_line.find(b",")].decode(), errors="coerce"
            )

            last_date = get_last_date(f)

            if pd.isna(start_date) or pd.isna(last_date):
                questionary.print(
                    f"WARNING: {file.name} - {invalid_dates}",
                    style=warning_color,
                )
                is_valid = False
                break

            if start_date > last_date:
                questionary.print(
                    f"WARNING: {file.name} - {invalid_date_format}",
                    style=warning_color,
                )
                is_valid = False
                break

        i += 1

    return is_valid


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
        "Provide folder path to OHLC data? Press Tab to autocomplete",
        only_directories=True,
        get_paths=lambda: [str(user)],
        validate=lambda fpath: Path(f"{user}/{fpath}").is_dir(),
    ).ask()


def ask_loader_name() -> str:
    loader_choice_list = ("Daily and Higher timeframe", "Intraday")

    loader_choice = questionary.select(
        "What timeframe is your OHLC data?",
        choices=loader_choice_list,
    ).ask()

    if loader_choice == loader_choice_list[0]:
        return "EODFileLoader"

    return "IEODFileLoader"


def ask_default_timeframe(loader: str) -> str:
    if loader == "EODFileLoader":
        # DEFAULT OHLC TIMEFRAME
        tf = questionary.select(
            "Select the timeframe of your OHLC Data?",
            choices=("Daily", "Weekly", "Monthly"),
        ).ask()

    else:
        tfs = ("1", "5", "10", "15", "25", "30", "60", "75", "125", "2h", "4h")

        # DEFAULT OHLC TIMEFRAME
        tf = questionary.select(
            "Select the timeframe of your OHLC Data?",
            choices=tfs,
        ).ask()

    return tf.lower()


def ask_watchlist(user: Path) -> Path:

    watchlist_path = questionary.path(
        """Provide the filepath to watchlist file?
    A text or CSV file with symbol names (one on each line).
    Press Tab to Autocomplete.""",
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

    questionary.print(
        f"# Filepaths are relative to: {user}\n", style=notice_color
    )

    # DEFAULT OR CUSTOM CONFIG
    if config_file.exists():
        config_choice_list = ["Edit user.json", "Create custom config"]

        config_choice = questionary.select(
            "What do you wish to do?",
            choices=config_choice_list,
        ).ask()

        # EDIT user.json
        if config_choice == config_choice_list[0]:
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
            config_file = DIR / "custom.json"

    # OHLC DATA FOLDER
    data_path = ask_default_source(user)

    print("Validating OHLC files")

    if not validate_ohlc_file(user / data_path):
        exit("Please correct OHLC file issues and try again.")

    questionary.print("✓ Passed validation", style=success_color)

    config["DATA_PATH"] = str(user / data_path)

    # OHLC LOADER
    loader_choice = ask_loader_name()
    config["LOADER"] = loader_choice

    config["DEFAULT_TF"] = ask_default_timeframe(loader_choice)

    # DEFAULT WATCHLIST FILE
    needs_watchlist = questionary.confirm(
        "Add a default watchlist to scan?"
    ).ask()

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
