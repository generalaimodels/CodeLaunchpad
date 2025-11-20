#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Educational system inventory script.

This script performs a broad, read-only inventory of a local machine and exports
the collected data to CSV files. The goal is to demonstrate clean Python design,
robust error handling, and cross-platform considerations while remaining strictly
within ethical and security-conscious boundaries.

IMPORTANT SECURITY NOTE
-----------------------
This script DELIBERATELY DOES NOT attempt to access or export any form of stored
credentials, including but not limited to:
    * Browser-saved passwords (Chrome, Firefox, Edge, etc.)
    * Operating-system credential stores (Windows Credential Manager, macOS Keychain,
      GNOME Keyring, etc.)
    * Wi-Fi passwords or other network authentication secrets
    * SSH private keys or agent sessions
    * Password-protected files or encrypted volumes
    * Any other sensitive authentication materials

Accessing or exporting such credentials programmatically can easily cross into
malware and credential-theft territory. For that reason, this script restricts
itself to non-credential system inventory only.

If you need to view your own saved passwords on a device you control, use the
official user interfaces exposed by:
    * Your web browser’s password manager UI
    * Your operating system’s credential manager / keychain UI
    * Your password manager application (e.g. KeePass, 1Password, Bitwarden, etc.)

These tools are designed with proper user interaction and consent flows, unlike
programmatic scraping, which is high-risk by nature.
"""

from __future__ import annotations

import csv
import datetime as dt
import getpass
import os
import platform
import shutil
import socket
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    # psutil is an excellent cross-platform library for system information.
    # It is optional: the script degrades gracefully if it is not installed.
    import psutil  # type: ignore[import-not-found]
except ImportError:
    psutil = None  # type: ignore[assignment]


try:
    # importlib.metadata is part of the standard library starting in Python 3.8.
    import importlib.metadata as importlib_metadata  # type: ignore[import-not-found]
except ImportError:
    importlib_metadata = None  # type: ignore[assignment]

try:
    # pkg_resources is provided by setuptools; it is a fallback for older Python
    # versions where importlib.metadata is unavailable.
    import pkg_resources  # type: ignore[import-not-found]
except ImportError:
    pkg_resources = None  # type: ignore[assignment]


# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------


@dataclass
class SystemOverviewRecord:
    """
    Aggregated one-row overview of core system properties, including OS-level
    identifiers and Python interpreter details.

    Most fields are represented as strings to simplify CSV serialization.
    """

    timestamp_utc: str
    username: str
    user_home: str
    working_directory: str
    hostname: str
    fqdn: str
    os_system: str
    os_release: str
    os_version: str
    os_platform: str
    machine: str
    processor: str
    architecture_bits: str
    architecture_linkage: str
    python_version: str
    python_implementation: str
    python_executable: str
    python_is_64bit: str
    psutil_available: str


@dataclass
class DiskRecord:
    """
    Description of a logical disk / filesystem as visible to the OS.
    Values are bytes for size-related fields, kept as strings in CSV output.
    """

    device: str
    mount_point: str
    file_system_type: str
    options: str
    total_bytes: str
    used_bytes: str
    free_bytes: str
    percent_used: str
    source: str  # Indicates whether data came from psutil or a fallback method


@dataclass
class MemoryRecord:
    """
    Representation of physical and swap memory statistics, where available.
    Values are bytes for size-related fields, kept as strings in CSV output.
    """

    type: str  # "virtual" or "swap"
    total_bytes: str
    available_bytes: str
    used_bytes: str
    free_bytes: str
    percent_used: str
    additional_info: str


@dataclass
class NetworkInterfaceRecord:
    """
    Representation of a single network address associated with a specific
    logical interface (e.g., eth0, Wi-Fi, loopback).
    """

    interface_name: str
    address_family: str
    address: str
    netmask: str
    broadcast: str
    ptp: str
    is_up: str
    mtu: str
    speed_mbps: str
    duplex: str
    mac_address: str


@dataclass
class PrimaryNetworkRecord:
    """
    Single-row representation of "primary" outbound IP addresses as observed
    by attempting to open a UDP socket.
    """

    primary_ipv4: str
    primary_ipv6: str


@dataclass
class ProcessRecord:
    """
    Representation of a running OS process. For portability and readability,
    we primarily rely on psutil attributes when available.
    """

    pid: str
    ppid: str
    name: str
    executable: str
    username: str
    status: str
    create_time_utc: str
    cmdline: str


@dataclass
class PythonPackageRecord:
    """
    Representation of a Python package installed in the current environment.
    """

    name: str
    version: str
    summary: str
    location: str
    installer: str
    metadata_source: str


@dataclass
class PythonEnvironmentRecord:
    """
    Representation of Python interpreter environment details beyond the high-level
    SystemOverviewRecord.
    """

    executable: str
    version: str
    implementation: str
    prefix: str
    base_prefix: str
    exec_prefix: str
    module_search_path: str


@dataclass
class EnvironmentVariableRecord:
    """
    Representation of a single environment variable. For security, values for
    variables that appear to hold secrets are redacted.
    """

    name: str
    value: str
    redacted: str
    reason: str


# --------------------------------------------------------------------------------------
# CSV utilities
# --------------------------------------------------------------------------------------


def ensure_output_directory(base_name: str = "system_inventory") -> Path:
    """
    Create a timestamped output directory within the current working directory.

    Using a timestamp avoids collisions between runs and makes each inventory
    snapshot self-contained and easy to archive.
    """
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    directory_name = f"{base_name}_{timestamp}_UTC"
    output_dir = Path.cwd() / directory_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_dicts_to_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """
    Serialize an iterable of mapping objects (e.g., dataclasses converted via asdict)
    into a CSV file. Fieldnames are inferred from the union of all row keys.

    Missing values for particular columns are written as empty strings to keep the
    CSV rectangular.
    """
    materialized_rows: List[Mapping[str, Any]] = list(rows)
    if not materialized_rows:
        # If there are no rows, there is nothing meaningful to write; we silently return.
        return

    # Compute the superset of all fieldnames across all rows.
    fieldnames: List[str] = sorted(
        {key for row in materialized_rows for key in row.keys()}
    )

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized_rows:
            safe_row: Dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                # Convert non-string values to string for CSV serialization.
                safe_row[key] = "" if value is None else str(value)
            writer.writerow(safe_row)


def dataclasses_to_csv(path: Path, objects: Iterable[Any]) -> None:
    """
    Convenience wrapper for writing an iterable of dataclass instances to CSV.
    """
    rows = (asdict(obj) for obj in objects)
    write_dicts_to_csv(path, rows)


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------


def safe_run_command(args: Sequence[str], timeout: int = 10) -> Tuple[int, str, str]:
    """
    Execute a local command and capture its exit code, stdout, and stderr.

    This helper is intentionally conservative: it avoids shell=True, uses a timeout,
    and returns text output. It is useful when falling back to OS utilities
    (e.g., df, ipconfig) on platforms where psutil is unavailable.

    Returns:
        (exit_code, stdout_str, stderr_str)
    """
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except Exception as exc:  # noqa: BLE001
        # In an educational setting, it is valuable to capture diagnostic information
        # while still allowing the script to continue execution.
        error_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        return 1, "", error_text


def get_primary_ip_addresses() -> PrimaryNetworkRecord:
    """
    Attempt to infer the primary outbound IPv4 and IPv6 addresses by opening
    UDP sockets to well-known public IPs. This does NOT send any traffic beyond
    the minimal low-level operations required to perform the connect() syscall.

    If this inference fails, empty strings are returned.
    """
    ipv4 = ""
    ipv6 = ""

    # IPv4 detection.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))  # Google Public DNS; reachable in most environments
            ipv4 = sock.getsockname()[0]
    except OSError:
        ipv4 = ""

    # IPv6 detection.
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as sock6:
            sock6.connect(("2001:4860:4860::8888", 80))  # Google Public DNS (IPv6)
            ipv6 = sock6.getsockname()[0]
    except OSError:
        ipv6 = ""

    return PrimaryNetworkRecord(primary_ipv4=ipv4, primary_ipv6=ipv6)


def is_probably_sensitive_env_name(name: str) -> bool:
    """
    Heuristically determine whether an environment variable name is likely to
    contain secrets (passwords, tokens, keys, etc.). This is intentionally
    biased toward caution.

    This heuristic is not perfect, but it demonstrates how logs / inventories
    can actively avoid exposing high-risk values.
    """
    upper_name = name.upper()
    sensitive_markers = [
        "PASS",
        "PWD",
        "SECRET",
        "TOKEN",
        "KEY",
        "AUTH",
        "CRED",
        "AWS_ACCESS",
        "AWS_SECRET",
        "API_KEY",
    ]
    return any(marker in upper_name for marker in sensitive_markers)


# --------------------------------------------------------------------------------------
# Collection functions
# --------------------------------------------------------------------------------------


def collect_system_overview() -> SystemOverviewRecord:
    """
    Collect high-level OS and Python interpreter properties.
    """
    timestamp_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    username = getpass.getuser()
    user_home = str(Path.home())
    working_directory = str(Path.cwd())

    hostname = socket.gethostname()
    fqdn = socket.getfqdn()

    os_system = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    os_platform = platform.platform(aliased=True, terse=False)
    machine = platform.machine()
    processor = platform.processor()

    arch_bits, arch_linkage = platform.architecture()
    python_version = platform.python_version()
    python_implementation = platform.python_implementation()
    python_executable = sys.executable or ""
    python_is_64bit = "true" if sys.maxsize > 2**32 else "false"

    psutil_available = "true" if psutil is not None else "false"

    return SystemOverviewRecord(
        timestamp_utc=timestamp_utc,
        username=username,
        user_home=user_home,
        working_directory=working_directory,
        hostname=hostname,
        fqdn=fqdn,
        os_system=os_system,
        os_release=os_release,
        os_version=os_version,
        os_platform=os_platform,
        machine=machine,
        processor=processor,
        architecture_bits=arch_bits,
        architecture_linkage=arch_linkage,
        python_version=python_version,
        python_implementation=python_implementation,
        python_executable=python_executable,
        python_is_64bit=python_is_64bit,
        psutil_available=psutil_available,
    )


def collect_disks() -> List[DiskRecord]:
    """
    Collect information about mounted disks / filesystems.

    Priority is given to psutil for portability. If psutil is unavailable,
    platform-specific fallbacks are used (shutil.disk_usage and simple
    drive enumeration).
    """
    records: List[DiskRecord] = []

    if psutil is not None:
        # psutil-based collection
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                total = str(usage.total)
                used = str(usage.used)
                free = str(usage.free)
                percent = f"{usage.percent:.2f}"
            except Exception:
                total = used = free = percent = ""
            record = DiskRecord(
                device=part.device,
                mount_point=part.mountpoint,
                file_system_type=part.fstype,
                options=part.opts,
                total_bytes=total,
                used_bytes=used,
                free_bytes=free,
                percent_used=percent,
                source="psutil",
            )
            records.append(record)
        return records

    # Fallback path when psutil is not installed.
    # We use a very simple drive enumeration strategy that will work "well enough"
    # for educational purposes.

    if os.name == "nt":
        # On Windows, logical drives are typically letters from C: onward.
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root_path = f"{letter}:\\"
            if not os.path.exists(root_path):
                continue
            try:
                usage = shutil.disk_usage(root_path)
                total = str(usage.total)
                used = str(usage.used)
                free = str(usage.free)
                percent = f"{(usage.used / usage.total) * 100:.2f}" if usage.total else ""
            except Exception:
                total = used = free = percent = ""
            record = DiskRecord(
                device=root_path,
                mount_point=root_path,
                file_system_type="",
                options="",
                total_bytes=total,
                used_bytes=used,
                free_bytes=free,
                percent_used=percent,
                source="fallback_windows",
            )
            records.append(record)
    else:
        # On POSIX systems, we can query df for a high-level overview.
        exit_code, stdout, _stderr = safe_run_command(["df", "-P", "-k"])
        if exit_code == 0:
            lines = stdout.strip().splitlines()
            # The first line is usually a header, so skip it.
            for line in lines[1:]:
                try:
                    # The POSIX df -P format is fairly regular: fields are spaced.
                    # Some systems may include different columns; we handle the most common.
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    device, blocks_kb, used_kb, avail_kb, capacity, mountpoint = parts[:6]
                    total_bytes = str(int(blocks_kb) * 1024)
                    used_bytes = str(int(used_kb) * 1024)
                    free_bytes = str(int(avail_kb) * 1024)
                    percent = capacity.strip().rstrip("%")
                except Exception:
                    device = mountpoint = ""
                    total_bytes = used_bytes = free_bytes = percent = ""
                record = DiskRecord(
                    device=device,
                    mount_point=mountpoint,
                    file_system_type="",
                    options="",
                    total_bytes=total_bytes,
                    used_bytes=used_bytes,
                    free_bytes=free_bytes,
                    percent_used=percent,
                    source="fallback_posix_df",
                )
                records.append(record)

    return records


def collect_memory() -> List[MemoryRecord]:
    """
    Collect information about physical and swap memory usage.

    This function prioritizes psutil for consistent cross-platform behavior.
    If psutil is unavailable, a very minimal and potentially approximate
    fallback is used, or the data is omitted entirely.
    """
    records: List[MemoryRecord] = []

    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            records.append(
                MemoryRecord(
                    type="virtual",
                    total_bytes=str(vm.total),
                    available_bytes=str(vm.available),
                    used_bytes=str(vm.used),
                    free_bytes=str(vm.free),
                    percent_used=f"{vm.percent:.2f}",
                    additional_info=f"active={getattr(vm, 'active', '')}, inactive={getattr(vm, 'inactive', '')}, buffers={getattr(vm, 'buffers', '')}, cached={getattr(vm, 'cached', '')}",
                )
            )
        except Exception:
            pass

        try:
            sm = psutil.swap_memory()
            records.append(
                MemoryRecord(
                    type="swap",
                    total_bytes=str(sm.total),
                    available_bytes=str(sm.free),
                    used_bytes=str(sm.used),
                    free_bytes=str(sm.free),
                    percent_used=f"{sm.percent:.2f}",
                    additional_info=f"sin={sm.sin}, sout={sm.sout}",
                )
            )
        except Exception:
            pass

        return records

    # psutil-less fallback: only very coarse information can be obtained portably.
    # Rather than embed complex, platform-specific ctypes logic, we simply avoid
    # guessing and instead leave the CSV empty or extremely minimal.

    if hasattr(os, "sysconf"):
        # POSIX-style sysconf can provide total physical memory on many systems.
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            total_bytes = page_size * phys_pages
            records.append(
                MemoryRecord(
                    type="virtual",
                    total_bytes=str(total_bytes),
                    available_bytes="",
                    used_bytes="",
                    free_bytes="",
                    percent_used="",
                    additional_info="Derived from POSIX sysconf; may be approximate.",
                )
            )
        except (ValueError, OSError):
            # sysconf can fail or be unavailable for certain keys.
            pass

    return records


def collect_network_interfaces() -> List[NetworkInterfaceRecord]:
    """
    Collect detailed information about network interfaces and their addresses.

    This relies heavily on psutil for portability. If psutil is unavailable,
    this function returns an empty list (network information can still be partially
    observed via the primary IP heuristic).
    """
    records: List[NetworkInterfaceRecord] = []

    if psutil is None:
        return records

    try:
        addrs_by_if = psutil.net_if_addrs()
        stats_by_if = psutil.net_if_stats()
    except Exception:
        return records

    for if_name, addr_list in addrs_by_if.items():
        stats = stats_by_if.get(if_name)
        is_up = ""
        mtu = ""
        speed = ""
        duplex = ""
        if stats is not None:
            is_up = "true" if stats.isup else "false"
            mtu = str(stats.mtu)
            speed = str(stats.speed)
            duplex = str(stats.duplex)

        # Track MAC address per interface (if present) to add explicitly.
        mac_address = ""
        for addr in addr_list:
            family_name = str(addr.family)
            if "AF_LINK" in family_name or "AF_PACKET" in family_name:
                mac_address = addr.address
                break

        for addr in addr_list:
            family = str(addr.family)
            record = NetworkInterfaceRecord(
                interface_name=if_name,
                address_family=family,
                address=addr.address or "",
                netmask=addr.netmask or "",
                broadcast=addr.broadcast or "",
                ptp=addr.ptp or "",
                is_up=is_up,
                mtu=mtu,
                speed_mbps=speed,
                duplex=duplex,
                mac_address=mac_address,
            )
            records.append(record)

    return records


def collect_processes() -> List[ProcessRecord]:
    """
    Collect information about currently running processes.

    We only attempt this if psutil is available; otherwise, the function
    returns an empty list. This decision keeps the implementation clear and
    avoids extensive parsing of platform-specific command outputs.
    """
    records: List[ProcessRecord] = []

    if psutil is None:
        return records

    # process_iter() is the most scalable way to enumerate processes via psutil.
    for proc in psutil.process_iter(
        attrs=[
            "pid",
            "ppid",
            "name",
            "exe",
            "username",
            "status",
            "create_time",
            "cmdline",
        ]
    ):
        try:
            info = proc.info
            pid = str(info.get("pid", ""))
            ppid = str(info.get("ppid", ""))
            name = info.get("name", "") or ""
            exe = info.get("exe", "") or ""
            username = info.get("username", "") or ""
            status = info.get("status", "") or ""
            create_time = info.get("create_time", None)
            if isinstance(create_time, (int, float)):
                dt_utc = dt.datetime.utcfromtimestamp(create_time).replace(
                    microsecond=0
                )
                create_time_utc = dt_utc.isoformat() + "Z"
            else:
                create_time_utc = ""
            cmdline_list = info.get("cmdline", [])
            if isinstance(cmdline_list, (list, tuple)):
                cmdline = " ".join(str(part) for part in cmdline_list)
            else:
                cmdline = ""
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            continue

        records.append(
            ProcessRecord(
                pid=pid,
                ppid=ppid,
                name=name,
                executable=exe,
                username=username,
                status=status,
                create_time_utc=create_time_utc,
                cmdline=cmdline,
            )
        )

    return records


def collect_python_packages() -> List[PythonPackageRecord]:
    """
    Collect information about Python packages installed in the current
    interpreter environment.

    Priority is given to importlib.metadata, with pkg_resources as a fallback.
    """
    records: List[PythonPackageRecord] = []

    if importlib_metadata is not None:
        try:
            for dist in importlib_metadata.distributions():
                try:
                    meta = dist.metadata
                except Exception:
                    meta = {}

                name = (
                    getattr(dist, "metadata", {}).get("Name")
                    if hasattr(dist, "metadata")
                    else None
                )
                if not name and hasattr(dist, "name"):
                    name = dist.name
                name = name or "UNKNOWN"

                version = getattr(dist, "version", "UNKNOWN")

                summary = ""
                try:
                    if meta:
                        summary = meta.get("Summary", "") or ""
                except Exception:
                    summary = ""

                try:
                    # dist.locate_file("") returns a path within the distribution;
                    # using parent directories gives a good approximation of the location.
                    location_path = dist.locate_file("")
                    location = str(location_path)
                except Exception:
                    location = ""

                installer = ""
                try:
                    if meta:
                        installer = meta.get("Installer", "") or ""
                except Exception:
                    installer = ""

                records.append(
                    PythonPackageRecord(
                        name=name,
                        version=str(version),
                        summary=summary,
                        location=location,
                        installer=installer,
                        metadata_source="importlib.metadata",
                    )
                )
        except Exception:
            # If anything goes wrong with importlib.metadata, we fall back to
            # pkg_resources below (if available).
            records.clear()

    if not records and pkg_resources is not None:
        # Fallback path using pkg_resources, which is somewhat slower but widely deployed.
        try:
            for dist in pkg_resources.working_set:
                name = dist.project_name or "UNKNOWN"
                version = dist.version or "UNKNOWN"
                summary = ""
                try:
                    summary = dist._get_metadata("PKG-INFO")  # type: ignore[attr-defined]
                    if summary:
                        # PKG-INFO is a multi-line metadata file; we do a very
                        # small amount of parsing for the Summary field.
                        for line in summary:
                            if line.startswith("Summary:"):
                                summary = line.split("Summary:", 1)[-1].strip()
                                break
                        else:
                            summary = ""
                except Exception:
                    summary = ""

                location = getattr(dist, "location", "") or ""
                installer = ""
                records.append(
                    PythonPackageRecord(
                        name=name,
                        version=str(version),
                        summary=str(summary),
                        location=location,
                        installer=installer,
                        metadata_source="pkg_resources",
                    )
                )
        except Exception:
            records.clear()

    return records


def collect_python_environment() -> PythonEnvironmentRecord:
    """
    Collect detailed Python interpreter environment information beyond high-level
    version and implementation, including sys.path and prefix data.
    """
    executable = sys.executable or ""
    version = sys.version.replace("\n", " ")
    implementation = platform.python_implementation()
    prefix = getattr(sys, "prefix", "")
    base_prefix = getattr(sys, "base_prefix", "")
    exec_prefix = getattr(sys, "exec_prefix", "")
    module_search_path = os.pathsep.join(sys.path)

    return PythonEnvironmentRecord(
        executable=executable,
        version=version,
        implementation=implementation,
        prefix=str(prefix),
        base_prefix=str(base_prefix),
        exec_prefix=str(exec_prefix),
        module_search_path=module_search_path,
    )


def collect_environment_variables() -> List[EnvironmentVariableRecord]:
    """
    Collect process environment variables, redacting values that are likely to
    contain secrets. This demonstrates responsible logging practices.

    NOTE:
    -----
    Even though environment variables are under user control, they often contain
    credentials (API keys, database passwords, cloud provider secrets). Dumping
    them unredacted would be risky.
    """
    records: List[EnvironmentVariableRecord] = []

    for name, value in os.environ.items():
        if is_probably_sensitive_env_name(name):
            records.append(
                EnvironmentVariableRecord(
                    name=name,
                    value="***REDACTED***",
                    redacted="true",
                    reason="Name heuristically matches sensitive credentials pattern.",
                )
            )
        else:
            records.append(
                EnvironmentVariableRecord(
                    name=name,
                    value=value,
                    redacted="false",
                    reason="",
                )
            )

    return records


def collect_additional_hardware_identifiers() -> Dict[str, str]:
    """
    Collect a limited set of hardware-related identifiers that are generally
    safe to expose for diagnostic purposes.

    NOTE:
    -----
    Some identifiers, such as MAC addresses, can be considered moderately
    sensitive in certain privacy contexts, but they are NOT authentication
    credentials. This function collects only a couple of such identifiers
    to illustrate how they can be obtained.

    For a more privacy-preserving inventory, reduce or omit these fields.
    """
    result: Dict[str, str] = {}

    # uuid.getnode() attempts to fetch a hardware MAC address or a random
    # 48-bit number if a MAC cannot be determined. We record both the raw
    # integer and a MAC-like formatted representation.
    try:
        node = uuid.getnode()
        result["uuid_getnode_raw"] = str(node)
        # Format as MAC-like hex representation (e.g., 01:23:45:67:89:ab).
        mac_like = ":".join(f"{(node >> ele) & 0xFF:02x}" for ele in range(40, -1, -8))
        result["uuid_getnode_mac_like"] = mac_like
    except Exception:
        result["uuid_getnode_raw"] = ""
        result["uuid_getnode_mac_like"] = ""

    return result


# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------


def main() -> None:
    """
    Orchestrate the full system inventory and export the results to CSV files.

    The following CSV files are produced in a timestamped directory:
        * system_overview.csv
        * disks.csv
        * memory.csv
        * network_interfaces.csv
        * network_primary.csv
        * processes.csv
        * python_packages.csv
        * python_environment.csv
        * environment_variables.csv
        * hardware_identifiers.csv

    Each file is self-describing based on its header row.

    As a crucial safety property, NO passwords or other credentials are
    programmatically extracted.
    """
    output_dir = ensure_output_directory(base_name="system_inventory")

    # 1. System overview
    overview_record = collect_system_overview()
    dataclasses_to_csv(output_dir / "system_overview.csv", [overview_record])

    # 2. Disk information
    disk_records = collect_disks()
    dataclasses_to_csv(output_dir / "disks.csv", disk_records)

    # 3. Memory information
    memory_records = collect_memory()
    dataclasses_to_csv(output_dir / "memory.csv", memory_records)

    # 4. Network interfaces
    net_if_records = collect_network_interfaces()
    dataclasses_to_csv(output_dir / "network_interfaces.csv", net_if_records)

    # 5. Primary network addresses
    primary_net_record = get_primary_ip_addresses()
    dataclasses_to_csv(output_dir / "network_primary.csv", [primary_net_record])

    # 6. Processes
    process_records = collect_processes()
    dataclasses_to_csv(output_dir / "processes.csv", process_records)

    # 7. Python packages
    python_package_records = collect_python_packages()
    dataclasses_to_csv(output_dir / "python_packages.csv", python_package_records)

    # 8. Python environment
    python_env_record = collect_python_environment()
    dataclasses_to_csv(output_dir / "python_environment.csv", [python_env_record])

    # 9. Environment variables (with redaction)
    env_var_records = collect_environment_variables()
    dataclasses_to_csv(output_dir / "environment_variables.csv", env_var_records)

    # 10. Additional hardware identifiers
    hw_identifiers = collect_additional_hardware_identifiers()
    write_dicts_to_csv(output_dir / "hardware_identifiers.csv", [hw_identifiers])

    # Optionally, you could print the output directory path here for convenience.
    # It is not strictly necessary for the script's core function.
    print(f"System inventory completed. Output directory: {output_dir}")


if __name__ == "__main__":
    main()