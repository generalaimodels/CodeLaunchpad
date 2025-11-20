#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module: os_command_inventory
Author: Educational Example
Description:
    Cross-platform script that:
      - Detects the current operating system
      - Enumerates executable commands discoverable via the PATH environment
      - Exports detailed information about each command into a CSV file

    All explanations and design rationale are embedded as comments and docstrings
    within this file for educational purposes.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Dict, Set, Optional, Type

from abc import ABC, abstractmethod


# -------------------------------------------------------------------------------------------------
# Data model
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandRecord:
    """
    Represents a single discovered command / executable on the system.

    Fields:
        os_name:
            Normalized name of the operating system on which discovery was performed
            (e.g., "Windows", "Linux", "Darwin", "Unknown").

        command_name:
            Base name of the command file (e.g., "python", "cmd.exe").
            This is the file name, not necessarily the shell alias or function name.

        absolute_path:
            Fully resolved absolute path to the underlying file that implements
            the command (e.g., "C:\\Windows\\System32\\cmd.exe" or "/usr/bin/python3").

        directory:
            Directory that contains the command (e.g., "/usr/bin").
            This is simply the parent directory of `absolute_path`.

        file_extension:
            File extension of the command (including the leading dot, e.g., ".exe",
            ".sh"), or an empty string if there is no extension.

        file_size_bytes:
            Size of the file in bytes, or None if the size could not be determined
            (e.g., due to permission errors or unusual filesystem semantics).

        last_modified_utc:
            UTC timestamp of the file's last modification time in ISO-8601 format
            (e.g., "2025-11-19T13:45:12.123456+00:00"), or None if unavailable.

        is_executable:
            Boolean flag indicating whether the file is considered executable
            by the enumeration logic for the given OS.

        discovery_source:
            Descriptor for how the command was discovered. For this script, the
            value is "PATH" because we scan directories listed in the PATH
            environment variable.

        discovery_error:
            Optional string describing any non-fatal error encountered while
            retrieving full metadata for this file. This allows capturing
            partial information when some fields cannot be populated.
    """

    os_name: str
    command_name: str
    absolute_path: str
    directory: str
    file_extension: str
    file_size_bytes: Optional[int]
    last_modified_utc: Optional[str]
    is_executable: bool
    discovery_source: str
    discovery_error: Optional[str] = None


# -------------------------------------------------------------------------------------------------
# OS detection utilities
# -------------------------------------------------------------------------------------------------


def detect_os_name() -> str:
    """
    Detects and returns a normalized name for the current operating system.

    Returns:
        A string such as "Windows", "Linux", "Darwin" (macOS), or "Unknown"
        if the platform cannot be reliably determined.
    """
    system_name = platform.system()
    if not system_name:
        return "Unknown"
    return system_name


# -------------------------------------------------------------------------------------------------
# Enumeration strategy abstraction
# -------------------------------------------------------------------------------------------------


class OSCommandEnumerator(ABC):
    """
    Abstract base class that defines a common interface for OS-specific
    command enumerators.

    Each concrete subclass is responsible for determining:
        - Which files should be considered "commands" on that OS
        - How to interpret the PATH and other OS-specific semantics
    """

    def __init__(self, os_name: str) -> None:
        """
        Initializes the enumerator with a normalized OS name.

        Args:
            os_name:
                Value from `detect_os_name`, used for tagging records and
                optionally tuning behavior.
        """
        self._os_name: str = os_name

    @abstractmethod
    def is_executable_file(self, path: Path) -> bool:
        """
        Determines whether the given file should be treated as a command
        for this operating system.

        Args:
            path:
                Path object representing a file candidate.

        Returns:
            True if the file is accepted as a command; False otherwise.
        """
        raise NotImplementedError

    def get_search_paths(self) -> List[Path]:
        """
        Parses the PATH environment variable into a list of existing directories.

        Behavior:
            - Splits PATH using `os.pathsep` (':' on POSIX, ';' on Windows).
            - Normalizes quotes and whitespace around each directory entry.
            - Filters out non-existing or non-directory entries.

        Returns:
            A list of Path objects corresponding to directories that will be
            scanned for command files.
        """
        raw_path = os.environ.get("PATH", "")
        search_paths: List[Path] = []

        if not raw_path:
            logging.warning("PATH environment variable is empty or undefined.")
            return search_paths

        for entry in raw_path.split(os.pathsep):
            cleaned = entry.strip().strip('"').strip("'")
            if not cleaned:
                continue
            directory = Path(cleaned)
            if directory.is_dir():
                search_paths.append(directory)
            else:
                logging.debug(
                    "Skipping PATH entry that is not a directory or does not exist: %s",
                    cleaned,
                )

        return search_paths

    def enumerate_commands(self) -> List[CommandRecord]:
        """
        Enumerates commands by scanning all directories in PATH.

        High-level algorithm:
            1. Collect all valid directory paths from PATH.
            2. For each directory:
                a. Iterate over child entries.
                b. For each file entry:
                    - Determine if it is a command via `is_executable_file`.
                    - Resolve its absolute path and gather metadata (size, mtime).
                    - Build a CommandRecord.
            3. Deduplicate entries by absolute path to avoid duplicates when
               the same directory is repeated in PATH or symlinks are present.

        Returns:
            A list of CommandRecord instances for all discovered commands.
        """
        search_paths = self.get_search_paths()
        logging.info("Using %d PATH directories for enumeration.", len(search_paths))

        records: List[CommandRecord] = []
        visited_paths: Set[str] = set()

        for directory in search_paths:
            try:
                entries = list(directory.iterdir())
            except PermissionError as exc:
                logging.warning(
                    "Permission denied while listing directory %s: %s",
                    directory,
                    exc,
                )
                continue
            except OSError as exc:
                logging.warning(
                    "Error while listing directory %s: %s",
                    directory,
                    exc,
                )
                continue

            for entry in entries:
                # Skip directories and non-files early to avoid unnecessary overhead
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    # Some exotic filesystems may raise errors on is_file; skip safely
                    continue

                # OS-specific check to determine if this file is a command
                try:
                    if not self.is_executable_file(entry):
                        continue
                except OSError:
                    # If the check itself fails (e.g., permission issues), treat as non-command
                    continue

                try:
                    resolved_path = entry.resolve(strict=False)
                except OSError:
                    # If resolution fails, fall back to the original path
                    resolved_path = entry.absolute()

                abs_path_str = str(resolved_path)

                # Deduplicate by absolute path
                if abs_path_str in visited_paths:
                    continue
                visited_paths.add(abs_path_str)

                # Attempt to collect metadata; failures are captured in discovery_error
                file_size: Optional[int] = None
                last_modified_utc: Optional[str] = None
                discovery_error: Optional[str] = None

                try:
                    stat_result = entry.stat()
                    file_size = stat_result.st_size

                    # Convert modification time to an ISO-8601 UTC timestamp
                    mtime_utc = datetime.fromtimestamp(
                        stat_result.st_mtime,
                        tz=timezone.utc,
                    )
                    last_modified_utc = mtime_utc.isoformat()
                except PermissionError as exc:
                    discovery_error = f"PermissionError during stat: {exc}"
                except OSError as exc:
                    discovery_error = f"OSError during stat: {exc}"

                record = CommandRecord(
                    os_name=self._os_name,
                    command_name=entry.name,
                    absolute_path=abs_path_str,
                    directory=str(entry.parent),
                    file_extension=entry.suffix or "",
                    file_size_bytes=file_size,
                    last_modified_utc=last_modified_utc,
                    is_executable=True,
                    discovery_source="PATH",
                    discovery_error=discovery_error,
                )

                records.append(record)

        logging.info("Discovered %d unique command files.", len(records))
        return records


# -------------------------------------------------------------------------------------------------
# POSIX implementation (Linux, macOS, etc.)
# -------------------------------------------------------------------------------------------------


class PosixCommandEnumerator(OSCommandEnumerator):
    """
    Command enumerator implementation for POSIX-compatible systems
    (e.g., Linux, macOS, other Unix-like platforms).

    Executable detection:
        - The file must be a regular file.
        - The current process must have execute permission on the file,
          as determined by `os.access(path, os.X_OK)`.
    """

    def is_executable_file(self, path: Path) -> bool:
        """
        POSIX-specific executable detection.

        Args:
            path:
                Path object pointing to a candidate file.

        Returns:
            True if `os.access(path, os.X_OK)` is True; False otherwise.
        """
        return os.access(path, os.X_OK)


# -------------------------------------------------------------------------------------------------
# Windows implementation
# -------------------------------------------------------------------------------------------------


class WindowsCommandEnumerator(OSCommandEnumerator):
    """
    Command enumerator implementation for Windows systems.

    Executable detection:
        - Windows does not rely on the UNIX executable bit. Instead, the
          PATHEXT environment variable specifies which file extensions should be
          treated as executable when encountered in PATH.
        - Typical PATHEXT values include:
              .COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC
    """

    def __init__(self, os_name: str) -> None:
        """
        Initializes the enumerator and parses the PATHEXT environment variable.

        Args:
            os_name:
                Normalized OS name (should be "Windows" for this implementation).
        """
        super().__init__(os_name=os_name)
        self._allowed_extensions = self._load_pathext()

    @staticmethod
    def _load_pathext() -> Set[str]:
        """
        Loads the PATHEXT environment variable into a normalized, lowercase set
        of extensions.

        Behavior:
            - If PATHEXT is not set, falls back to a common default list of
              executable extensions used by Windows.

        Returns:
            A set of lowercase extensions, each including the leading dot.
        """
        default_extensions = [
            ".com",
            ".exe",
            ".bat",
            ".cmd",
            ".vbs",
            ".vbe",
            ".js",
            ".jse",
            ".wsf",
            ".wsh",
            ".msc",
        ]

        raw_pathext = os.environ.get("PATHEXT", "")
        if not raw_pathext:
            return set(default_extensions)

        extensions: Set[str] = set()
        for entry in raw_pathext.split(os.pathsep):
            cleaned = entry.strip().lower()
            if cleaned:
                extensions.add(cleaned)

        if not extensions:
            # In case PATHEXT was defined but effectively empty
            extensions.update(default_extensions)

        return extensions

    def is_executable_file(self, path: Path) -> bool:
        """
        Windows-specific executable detection.

        Args:
            path:
                Path object pointing to a candidate file.

        Returns:
            True if the file's extension is in the PATHEXT-derived set;
            False otherwise.
        """
        suffix = path.suffix.lower()
        return suffix in self._allowed_extensions


# -------------------------------------------------------------------------------------------------
# Factory for creating the appropriate enumerator
# -------------------------------------------------------------------------------------------------


def create_enumerator(os_name: str) -> OSCommandEnumerator:
    """
    Factory function that instantiates an OS-specific command enumerator
    based on the detected operating system name.

    Args:
        os_name:
            Result of `detect_os_name()`, such as "Windows", "Linux", or "Darwin".

    Returns:
        An instance of a concrete subclass of OSCommandEnumerator.
    """
    if os_name == "Windows":
        return WindowsCommandEnumerator(os_name=os_name)

    # All non-Windows platforms are treated as POSIX-compatible for the purpose
    # of executable detection logic. This includes Linux and macOS ("Darwin").
    return PosixCommandEnumerator(os_name=os_name)


# -------------------------------------------------------------------------------------------------
# CSV export utilities
# -------------------------------------------------------------------------------------------------


def write_commands_to_csv(
    records: Iterable[CommandRecord],
    output_path: Path,
    encoding: str = "utf-8",
) -> None:
    """
    Serializes a collection of CommandRecord instances into a CSV file.

    Args:
        records:
            Iterable of CommandRecord objects to be written.

        output_path:
            Path object specifying where the CSV file will be created or overwritten.

        encoding:
            Text encoding for the generated CSV file (defaults to UTF-8).

    Behavior:
        - Writes a header row containing column names.
        - Each subsequent row corresponds to one CommandRecord.
        - The CSV format is intentionally simple and compatible with most
          spreadsheet tools.
    """
    # Determine the field order explicitly to ensure consistent CSV structure
    fieldnames = [
        "os_name",
        "command_name",
        "absolute_path",
        "directory",
        "file_extension",
        "file_size_bytes",
        "last_modified_utc",
        "is_executable",
        "discovery_source",
        "discovery_error",
    ]

    # Ensure the parent directory exists if a nested path is provided
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", encoding=encoding, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            row: Dict[str, object] = asdict(record)
            writer.writerow(row)

    logging.info("CSV export completed. Output file: %s", output_path)


# -------------------------------------------------------------------------------------------------
# Argument parsing and main entry point
# -------------------------------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parses command-line arguments for this script.

    Args:
        argv:
            Optional list of arguments, typically `sys.argv[1:]`. If None, the
            current process arguments are used.

    Returns:
        A populated argparse.Namespace instance with parsed options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate OS-specific command files discoverable via PATH and "
            "export detailed metadata to a CSV file."
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="os_commands_inventory.csv",
        help=(
            "Path to the CSV file to create (default: ./os_commands_inventory.csv). "
            "Intermediate directories will be created as needed."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging to aid in understanding script behavior.",
    )

    return parser.parse_args(argv)


def configure_logging(debug: bool) -> None:
    """
    Configures the Python logging system according to user preference.

    Args:
        debug:
            If True, sets logging level to DEBUG; otherwise, INFO.
    """
    log_level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for module execution.

    High-level workflow:
        1. Parse CLI arguments and configure logging.
        2. Detect the operating system.
        3. Instantiate the appropriate OSCommandEnumerator.
        4. Enumerate commands from PATH.
        5. Write results to the requested CSV file.
    """
    args = parse_args(argv)
    configure_logging(debug=args.debug)

    os_name = detect_os_name()
    logging.info("Detected operating system: %s", os_name)

    enumerator = create_enumerator(os_name=os_name)
    records = enumerator.enumerate_commands()

    output_path = Path(args.output).expanduser().resolve()
    write_commands_to_csv(records=records, output_path=output_path)


if __name__ == "__main__":
    main(sys.argv[1:])