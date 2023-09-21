from __future__ import annotations

import argparse
import json
import os
import pty
import re
import subprocess
import sys
from typing import List

# NOTE
# https://clang.llvm.org/extra/doxygen/run-clang-tidy_8py_source.html


def should_ignore_file(file: str, ignore_words: str) -> bool:
    """Check if file should be ignored based on word list.

    Args:
        ignore_word_list: Word list to be ignored.

    Returns:
        If file includes one of word in list, return True, Otherwise, return False.
    """
    # If ignore_words are empty, everything should not be ignored.
    if not ignore_words:
        return False

    result = re.search(rf".*({ignore_words}).*", file)
    return False if result is None else True


def is_target_file(file: str, target_files: List(str)) -> bool:
    """Check if specified file is one of target files

    Args
        file: File name to be check if it is one of target files
        target_files: List of target files
    """
    if len(target_files) == 0:
        return True

    file_in = [item for item in target_files if item in file]
    return True if len(file_in) > 0 else False


def exec_clang_tidy(
    file: str, header_filter: str, will_fix: bool, will_use_subprocee=True
):
    """Execute clang_tidy.

    Args:
        file: File for clagn_tidy.
        compile_options: Compile option list.
        header_filter: Header filter to be passed to clang_tidy "--header-filter" option.
    """
    clang_tidy_cmd = ["clang-tidy-12", file]

    if will_fix:
        clang_tidy_cmd.append("--fix-errors")

    if header_filter:
        clang_tidy_cmd.append(f"--header-filter={header_filter}")

    if will_use_subprocee:
        proc = subprocess.Popen(
            clang_tidy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # NOTE
        # https://stackoverflow.com/questions/53196795/why-are-python-3-line-breaks-n-not-working-with-print-and-subprocess-popen-stdo
        print(proc.stdout.read().decode())

        proc.communicate()
    else:
        # NOTE
        # https://stackoverflow.com/questions/21442360/issuing-commands-to-psuedo-shells-pty
        pty.spawn(clang_tidy_cmd)


def exec_clang_tidy_by_compile_commands_json(
    compile_commands_json: str,
    ignore_words: str,
    header_filter: str,
    will_fix: bool,
    target_files: List[str],
):
    """Execute clang_tidy based on compile_commands.json file.

    Args:
        compile_commands_json: Path to compile_commands.json file.
        ignore_word_list: Word list to be ignored.
        header_filter: Header filter to be passed to clang_tidy "--header-filter" option.
        target_file: Target file. If this is specified, execute clang_tidy for only target file.
    """
    with open(compile_commands_json, mode="r") as f:
        compilation_db = json.load(f)
        if compilation_db is None:
            # TODO
            # print error
            pass
        for item in compilation_db:
            file = item["file"]
            should_ignore = should_ignore_file(file, ignore_words)
            if should_ignore:
                continue

            is_target = is_target_file(file, target_files)

            if is_target:
                exec_clang_tidy(file, header_filter, will_fix)


def main():
    # NOTE:
    # e.g.
    # python3 ./scripts/run_clang_tidy.py -i 3rdparty imgui unittest --header_filter "${PWD}/src/" -t accelerator.cpp  # noqa
    parser = argparse.ArgumentParser(description="Run clang-tidy")
    parser.add_argument(
        "-i",
        "--ignore_words",
        nargs="*",
        type=str,
        help="List to ignore. If file path include specified work, that file is ignored",
        default=[],
    )
    parser.add_argument(
        "--header_filter", type=str, help="Header filter for clang-tidy", default=None
    )
    parser.add_argument(
        "-t",
        "--target_files",
        nargs="*",
        type=str,
        help="Specific target file to run clang-tidy",
        default=[],
    )
    parser.add_argument("-f", "--fix", action="store_true", help="", default=False)
    args = parser.parse_args()

    # Compile ignore words.
    ignore_words = ""
    for word in args.ignore_words:
        if not ignore_words:
            ignore_words += word
        else:
            ignore_words += f"|{word}"

    COMPILE_COMMANDS_JSON = "compile_commands.json"

    # compile_commands.json need to be genareted beforehand.
    if not os.path.exists(COMPILE_COMMANDS_JSON):
        print(f"No {COMPILE_COMMANDS_JSON}. Need to generate beforehand")
        sys.exit(1)

    try:
        exec_clang_tidy_by_compile_commands_json(
            compile_commands_json=COMPILE_COMMANDS_JSON,
            ignore_words=ignore_words,
            header_filter=args.header_filter,
            will_fix=args.fix,
            target_files=args.target_files,
        )
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
