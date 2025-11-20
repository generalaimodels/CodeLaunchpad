#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local Wi-Fi / LAN device discovery script (improved version).

This script attempts to detect all devices that are currently reachable on the
same IPv4 network as the primary interface used for internet access (typically
your Wi-Fi adapter on a laptop).

Major improvements over the basic version:
- Silent and robust Scapy import (no noisy "No libpcap" warnings by default).
- Configurable scanning parameters via command-line arguments:
    * Scan method: auto / arp / ping.
    * Adjustable ICMP timeout and concurrency (thread count).
    * Optional reverse-DNS resolution (for faster silent scans).
    * Optional manual network override instead of auto-detected subnet.
- Safer guard for very large networks, controlled via CLI.
- Clean separation of responsibilities and consistently documented helpers.

Key features:
- Auto-detects the primary local IP address (outbound route based).
- Infers the local IPv4 subnet (CIDR) using platform-specific tools.
- Uses Scapy for ARP-based scanning when possible and requested.
- Falls back to ICMP ping scanning plus parsing the system ARP table.
- Collects IP, MAC address, and reverse-DNS hostname (where available).
- Presents a neatly formatted table of discovered devices.

IMPORTANT:
- Use this script only on networks you own or are explicitly authorized to
  audit or analyze. Unauthorized network scanning may be illegal or violate
  network policies.
- ARP-based scanning typically requires elevated privileges (root/Administrator).
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from time import monotonic
from typing import Dict, Iterable, List, Optional

# -----------------------------------------------------------------------------
# Best-effort, silent Scapy import
# -----------------------------------------------------------------------------
# Scapy is optional; if unavailable, the script will gracefully fall back
# to pure standard-library / OS-tool based scanning.
# On some platforms (especially Windows without WinPcap/NPcap), Scapy prints
# "No libpcap provider available" during import; we hide that noise here.
os.environ.setdefault("SCAPY_SUPPRESS_NO_PCAP", "1")
try:
    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        import scapy.all as scapy  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001 - any failure simply disables Scapy usage
    scapy = None

# Set a reasonably small default timeout for all new sockets (e.g., reverse DNS).
# This helps avoid the script hanging on slow or non-responsive DNS queries.
socket.setdefaulttimeout(1.0)


@dataclass
class DetectedDevice:
    """
    Represents a device discovered on the local network.

    Fields:
        ip:
            IPv4 address of the device as a string.
        mac:
            MAC address of the device if known, otherwise None.
        hostname:
            Reverse-DNS hostname if resolvable, otherwise None.
        is_self:
            True if this IP is the current host's primary IP.
    """

    ip: str
    mac: Optional[str] = None
    hostname: Optional[str] = None
    is_self: bool = False


@dataclass
class ScannerOptions:
    """
    Configuration options for LocalNetworkScanner.

    Fields:
        icmp_timeout:
            Timeout in seconds for each ICMP (ping) probe.
        arp_timeout:
            Timeout in seconds for ARP-based probing when using Scapy.
        max_workers:
            Maximum concurrent worker threads for parallel pinging.
        prefer_arp:
            If True, attempt ARP-based scan first when Scapy is available.
        enable_reverse_dns:
            If True, attempt reverse-DNS lookups to resolve hostnames.
        max_hosts:
            Safety limit for maximum number of IPs to scan within a network.
        override_network:
            Optional IPv4 network string (CIDR) to scan instead of auto-inferred
            one, e.g. "192.168.1.0/24". If provided, it must contain the
            primary local IP or an error will be raised.
    """

    icmp_timeout: float = 0.75
    arp_timeout: float = 2.0
    max_workers: int = 128
    prefer_arp: bool = True
    enable_reverse_dns: bool = True
    max_hosts: int = 4096
    override_network: Optional[str] = None


class LocalNetworkScanner:
    """
    High-level orchestrator for local IPv4 network discovery.

    Responsibilities:
    - Local IP discovery.
    - Subnet inference or override handling.
    - ARP-based and/or ICMP-based host discovery.
    - ARP table parsing.
    - Result aggregation and presentation.
    """

    def __init__(self, options: Optional[ScannerOptions] = None) -> None:
        """
        Initialize the scanner with the provided options.

        Args:
            options:
                Optional ScannerOptions instance. If None, default options
                defined in ScannerOptions() are used.
        """
        if options is None:
            options = ScannerOptions()

        # Basic validation for core numeric parameters.
        if options.icmp_timeout <= 0.0:
            raise ValueError("icmp_timeout must be positive")
        if options.arp_timeout <= 0.0:
            raise ValueError("arp_timeout must be positive")
        if options.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if options.max_hosts <= 0:
            raise ValueError("max_hosts must be positive")

        self.options = options

        # Pre-parsed override network, if supplied.
        self._override_network: Optional[ipaddress.IPv4Network] = None
        if self.options.override_network:
            # strict=False ensures we accept host addresses and normalize them.
            self._override_network = ipaddress.IPv4Network(
                self.options.override_network,
                strict=False,
            )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def scan(self) -> List[DetectedDevice]:
        """
        Execute a complete network scan and return the list of detected devices.

        High-level algorithm:
        1. Determine the primary local IPv4 address.
        2. Infer (or validate override) the IPv4 network containing that IP.
        3. Optionally limit scanning on very large networks.
        4. Attempt ARP-based scanning if enabled and Scapy is available.
           - If ARP-based scan fails or is disabled, fall back to ICMP ping scan.
        5. After ping scan, read ARP table to obtain MAC addresses where possible.
        6. Aggregate results: IP, MAC, reverse-DNS hostname (optional),
           and "is self" flag.

        Returns:
            A list of DetectedDevice instances sorted by IP address.
        """
        primary_ip = self._get_primary_ipv4()
        network = self._determine_network(primary_ip)
        self._guard_against_large_networks(network)

        # Step 1: ARP-based detection (fastest and most accurate on local LAN).
        ip_mac_map: Dict[str, str] = {}
        if self.options.prefer_arp and scapy is not None:
            ip_mac_map = self._arp_scan_with_scapy(network)

        # Step 2: ICMP-based detection fallback if ARP scanning was not used
        # or did not yield any results.
        if not ip_mac_map:
            # Do not ping the primary IP; we already know it is alive.
            hosts = [
                host
                for host in network.hosts()
                if str(host) != primary_ip
            ]
            reachable_ips = self._discover_live_hosts_via_ping(hosts)

            # ARP cache should now contain entries for many of these hosts.
            arp_table = self._read_arp_table()

            # Build IP -> MAC map using ARP table for reachable hosts.
            ip_mac_map = {ip: arp_table.get(ip, "") for ip in reachable_ips}
            # If MAC is unknown for some IPs, store empty string; we normalize
            # to None later when constructing DetectedDevice instances.

        # Step 3: Convert results into DetectedDevice structures and enrich them
        # with reverse-DNS hostnames where requested.
        devices = self._build_device_list_from_results(
            ip_mac_map=ip_mac_map,
            primary_ip=primary_ip,
        )

        # Sort devices by numeric IPv4 order for stable presentation.
        devices.sort(key=lambda d: ipaddress.IPv4Address(d.ip))
        return devices

    # -------------------------------------------------------------------------
    # Primary IP and network detection
    # -------------------------------------------------------------------------

    def _get_primary_ipv4(self) -> str:
        """
        Determine the primary local IPv4 address used for outbound connectivity.

        Technique:
        - Create a UDP socket and "connect" it to a well-known external address
          (e.g., 8.8.8.8 on port 80). This does not send any packets, but
          forces the OS to select a local interface/IP appropriate for routing.
        - Query the socket's local endpoint address to discover the IP.

        Returns:
            A string with the local IPv4 address.

        Raises:
            RuntimeError:
                If the primary IP cannot be determined or is invalid.
        """
        test_endpoint = ("8.8.8.8", 80)  # Arbitrary; no real traffic is sent.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(test_endpoint)
                local_ip = sock.getsockname()[0]
        except OSError as exc:
            raise RuntimeError(
                f"Failed to determine primary IPv4 address: {exc}",
            ) from exc

        # Ensure the string is valid IPv4.
        try:
            ipaddress.IPv4Address(local_ip)
        except ipaddress.AddressValueError as exc:
            raise RuntimeError(
                f"Determined local IP '{local_ip}' is not a valid IPv4 address",
            ) from exc

        return local_ip

    def _determine_network(self, primary_ip: str) -> ipaddress.IPv4Network:
        """
        Determine which IPv4 network should be scanned.

        Logic:
        - If an override network was configured:
            * Ensure it is IPv4.
            * Ensure it contains the primary IP; if not, abort with an error
              (to avoid accidentally scanning unrelated networks).
        - Otherwise:
            * Infer subnet from OS configuration (ip/ipconfig/ifconfig).
            * Fallback to /24 if that fails.

        Args:
            primary_ip:
                The primary local IPv4 address as a string.

        Returns:
            An ipaddress.IPv4Network instance representing the scan range.
        """
        if self._override_network is not None:
            network = self._override_network
            if not isinstance(network, ipaddress.IPv4Network):
                raise ValueError("Only IPv4 networks are supported")
            if ipaddress.IPv4Address(primary_ip) not in network:
                raise ValueError(
                    f"Override network {network} does not contain "
                    f"the primary IP {primary_ip}",
                )
            return network

        # When override is not supplied, infer the network based on primary IP.
        return self._infer_network_for_ip(primary_ip)

    def _infer_network_for_ip(self, ip: str) -> ipaddress.IPv4Network:
        """
        Infer the IPv4 network (CIDR) the given IP belongs to.

        Strategy:
        1. On Windows:
           - Run 'ipconfig' and find the block containing the given IP.
           - Parse the 'Subnet Mask' line.
        2. On Linux / macOS / BSD:
           - Prefer 'ip -o -f inet addr show' if the 'ip' tool exists.
           - Otherwise, parse 'ifconfig' output.
        3. If all attempts fail, conservatively assume a /24 network.

        Args:
            ip:
                The local IPv4 address as a string.

        Returns:
            An ipaddress.IPv4Network instance that includes the given IP.
        """
        system = platform.system().lower()
        mask_str: Optional[str] = None
        prefix_len: Optional[int] = None

        if system == "windows":
            mask_str = self._get_netmask_from_ipconfig(ip)
        else:
            # UNIX-like platforms: try 'ip' first, then 'ifconfig'.
            prefix_len = self._get_prefixlen_from_ip_command(ip)
            if prefix_len is None:
                mask_str = self._get_netmask_from_ifconfig(ip)

        # Convert any subnet mask string to prefix length.
        if mask_str and prefix_len is None:
            prefix_len = self._netmask_to_prefixlen(mask_str)

        # If everything failed, fall back to /24 (common home network).
        if prefix_len is None:
            prefix_len = 24

        network = ipaddress.IPv4Network(
            f"{ip}/{prefix_len}",
            strict=False,
        )
        return network

    def _get_netmask_from_ipconfig(self, ip: str) -> Optional[str]:
        """
        Extract the subnet mask for a given IP by parsing 'ipconfig' on Windows.

        Parsing approach:
        - Run 'ipconfig' without arguments.
        - Locate the line containing the given IPv4 address.
        - Search a few subsequent lines for 'Subnet Mask' and capture its value.

        Args:
            ip:
                The IPv4 address whose subnet mask we are trying to infer.

        Returns:
            The subnet mask as a dotted-quad string, or None if not found.
        """
        try:
            output = subprocess.check_output(
                ["ipconfig"],
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except (OSError, subprocess.SubprocessError):
            return None

        lines = output.splitlines()
        target_indices: List[int] = []

        for idx, line in enumerate(lines):
            if ip in line:
                target_indices.append(idx)

        if not target_indices:
            return None

        subnet_mask_regex = re.compile(
            r"Subnet\s+Mask[^\:]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})",
            re.IGNORECASE,
        )
        for idx in target_indices:
            # Scan a small window following the IP line.
            for j in range(idx, min(idx + 10, len(lines))):
                match = subnet_mask_regex.search(lines[j])
                if match:
                    return match.group(1)

        return None

    def _get_prefixlen_from_ip_command(self, ip: str) -> Optional[int]:
        """
        On UNIX-like systems, attempt to get CIDR prefix length via 'ip' tool.

        Example 'ip -o -f inet addr show' line:
            3: wlp0s20f3 inet 192.168.1.50/24 brd 192.168.1.255 scope global ...

        Args:
            ip:
                The IPv4 address whose prefix length is being determined.

        Returns:
            The prefix length (e.g., 24 for /24) or None if not found or 'ip'
            is unavailable.
        """
        ip_exe = shutil.which("ip")
        if not ip_exe:
            return None

        try:
            output = subprocess.check_output(
                [ip_exe, "-o", "-f", "inet", "addr", "show"],
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except (OSError, subprocess.SubprocessError):
            return None

        pattern = re.compile(
            rf"\binet\s+{re.escape(ip)}/(\d+)\b",
            re.IGNORECASE,
        )

        for line in output.splitlines():
            match = pattern.search(line)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue

        return None

    def _get_netmask_from_ifconfig(self, ip: str) -> Optional[str]:
        """
        Parse 'ifconfig' output on UNIX-like systems to find the subnet mask.

        Example outputs:
        - Linux (modern):
            inet 192.168.1.50  netmask 255.255.255.0  broadcast 192.168.1.255
        - macOS / BSD:
            inet 192.168.1.50 netmask 0xffffff00 broadcast 192.168.1.255

        Args:
            ip:
                The IP whose subnet mask we want to recover.

        Returns:
            The subnet mask as dotted-quad (e.g., '255.255.255.0'),
            or None if not found.
        """
        ifconfig_exe = shutil.which("ifconfig")
        if not ifconfig_exe:
            return None

        try:
            output = subprocess.check_output(
                [ifconfig_exe],
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except (OSError, subprocess.SubprocessError):
            return None

        lines = output.splitlines()
        ip_pattern = re.compile(rf"\binet\s+{re.escape(ip)}\b")

        candidate_indices: List[int] = [
            idx for idx, line in enumerate(lines) if ip_pattern.search(line)
        ]
        if not candidate_indices:
            return None

        dotted_mask_re = re.compile(
            r"\bnetmask\s+([0-9]{1,3}(?:\.[0-9]{1,3}){3})\b",
            re.IGNORECASE,
        )
        hex_mask_re = re.compile(
            r"\bnetmask\s+0x([0-9a-fA-F]{8})\b",
            re.IGNORECASE,
        )

        for idx in candidate_indices:
            # Search around the line where the IP was found.
            for j in range(max(0, idx - 2), min(idx + 3, len(lines))):
                line = lines[j]
                dotted_match = dotted_mask_re.search(line)
                if dotted_match:
                    return dotted_match.group(1)

                hex_match = hex_mask_re.search(line)
                if hex_match:
                    hex_value = hex_match.group(1)
                    try:
                        mask_int = int(hex_value, 16)
                    except ValueError:
                        continue
                    # Convert integer mask to dotted-quad string.
                    bytes_tuple = [
                        (mask_int >> shift) & 0xFF
                        for shift in (24, 16, 8, 0)
                    ]
                    dotted = ".".join(str(b) for b in bytes_tuple)
                    return dotted

        return None

    def _netmask_to_prefixlen(self, mask: str) -> Optional[int]:
        """
        Convert a dotted-quad subnet mask (e.g. '255.255.255.0') to prefix length.

        Validation:
        - Ensures that the binary representation is contiguous (e.g., '11111111'
          followed by '0's only). Non-contiguous masks are considered invalid.

        Args:
            mask:
                Subnet mask as dotted-quad string.

        Returns:
            Prefix length (0-32) if the mask is valid, otherwise None.
        """
        octets = mask.split(".")
        if len(octets) != 4:
            return None

        try:
            bytes_list = [int(octet) for octet in octets]
        except ValueError:
            return None

        if any(not (0 <= b <= 255) for b in bytes_list):
            return None

        bin_str = "".join(f"{b:08b}" for b in bytes_list)

        # Valid masks must be contiguous ones followed by zeros.
        if "01" in bin_str:
            return None

        return bin_str.count("1")

    # -------------------------------------------------------------------------
    # Network size guard
    # -------------------------------------------------------------------------

    def _guard_against_large_networks(
        self,
        network: ipaddress.IPv4Network,
    ) -> None:
        """
        Optionally guard against accidentally scanning very large networks.

        On typical home Wi-Fi routers, networks are /24 or sometimes /23.
        Larger networks may indicate enterprise or ISP-level ranges where a
        brute-force scan can be both slow and potentially unwelcome.

        Behavior:
        - If the number of addresses exceeds 'options.max_hosts', prompt the
          user for confirmation on stdin.
        - If stdin is not interactive (e.g., running in automation), the scan
          is aborted to avoid unintended behavior.

        Args:
            network:
                The network about to be scanned.

        Raises:
            SystemExit:
                If the user declines to scan or if stdin is non-interactive
                and the network is too large.
        """
        total_addresses = network.num_addresses
        max_hosts = self.options.max_hosts
        if total_addresses <= max_hosts:
            return

        # If not running in an interactive terminal, abort.
        if not sys.stdin.isatty():
            print(
                f"[ABORT] Network {network} has {total_addresses} addresses, "
                f"which exceeds the safety limit ({max_hosts}) in "
                "non-interactive mode.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        print(
            f"[WARN] Network {network} has {total_addresses} addresses, "
            f"which exceeds the safety soft limit of {max_hosts}.",
            file=sys.stderr,
        )
        print(
            "Scanning such a large range may be slow and could be "
            "inappropriate on some environments.",
            file=sys.stderr,
        )
        answer = input(
            "Do you still want to proceed with scanning this network? [y/N]: ",
        ).strip().lower()

        if answer not in ("y", "yes"):
            print("[INFO] Scan aborted by user.", file=sys.stderr)
            raise SystemExit(1)

    # -------------------------------------------------------------------------
    # ARP scanning (Scapy-based, if available)
    # -------------------------------------------------------------------------

    def _arp_scan_with_scapy(
        self,
        network: ipaddress.IPv4Network,
    ) -> Dict[str, str]:
        """
        Perform ARP-based discovery using Scapy, if available and permitted.

        ARP scan advantages:
        - Very fast for local networks.
        - Directly yields IP-to-MAC mappings.
        - Does not depend on ping or ICMP reachability.

        Requirements:
        - Scapy must be installed (pip install scapy).
        - Root/Administrator privileges are typically required to send
          raw Ethernet frames.
        - On Windows, a proper packet capture driver (NPcap/WinPcap) is required
          for layer-2 operations; otherwise this method will fail.

        Args:
            network:
                The IPv4 network to scan.

        Returns:
            A mapping of IP string -> MAC address string.
            Returns an empty dict if Scapy is unavailable or scanning fails.
        """
        if scapy is None:
            return {}

        # Best-effort detection of missing L2 capabilities: if Scapy's conf
        # has no L2socket, ARP over Ethernet is unlikely to work.
        try:
            if getattr(scapy.conf, "L2socket", None) is None:
                # Returning early avoids confusing runtime exceptions.
                return {}
        except Exception:  # noqa: BLE001
            # If inspection fails, still try and rely on exception handling.
            pass

        try:
            target = str(network)
            arp_request = scapy.ARP(pdst=target)
            ether = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
            packet = ether / arp_request

            start_time = monotonic()
            answered_list, _ = scapy.srp(
                packet,
                timeout=self.options.arp_timeout,
                verbose=False,
            )
            duration = monotonic() - start_time

            ip_mac_map: Dict[str, str] = {}
            for _sent_pkt, recv_pkt in answered_list:
                ip_addr = recv_pkt.psrc
                mac_addr = recv_pkt.hwsrc
                ip_mac_map[ip_addr] = mac_addr

            # Log high-level stats to stderr to aid understanding.
            sys.stderr.write(
                f"[INFO] ARP scan (Scapy) completed in {duration:.2f}s, "
                f"found {len(ip_mac_map)} hosts.\n",
            )

            return ip_mac_map

        except PermissionError:
            # Insufficient privileges for raw sockets.
            sys.stderr.write(
                "[WARN] ARP scan via Scapy failed due to insufficient "
                "permissions; falling back to ICMP ping scan.\n",
            )
        except RuntimeError as exc:
            # Common case on Windows without WinPcap/NPcap.
            sys.stderr.write(
                "[WARN] ARP scan via Scapy is not available at layer 2 "
                f"({exc}); falling back to ICMP ping scan.\n",
            )
        except Exception as exc:  # noqa: BLE001
            # Any unexpected error causes a fallback to ping scan.
            sys.stderr.write(
                f"[WARN] ARP scan via Scapy failed ({exc!r}); "
                "falling back to ICMP ping scan.\n",
            )

        return {}

    # -------------------------------------------------------------------------
    # ICMP ping-based scanning
    # -------------------------------------------------------------------------

    def _discover_live_hosts_via_ping(
        self,
        hosts: Iterable[ipaddress.IPv4Address],
    ) -> List[str]:
        """
        Discover live hosts by issuing ICMP echo requests ("ping") in parallel.

        Implementation:
        - Uses OS 'ping' command via subprocess for portability.
        - Executes pings concurrently using ThreadPoolExecutor.
        - Considers a host "live" if ping exit code is 0.

        Args:
            hosts:
                Iterable of IPv4Address objects representing potential hosts.

        Returns:
            A list of reachable IP addresses (as strings).
        """
        host_strings = [str(h) for h in hosts]
        live_hosts: List[str] = []

        if not host_strings:
            return live_hosts

        sys.stderr.write(
            f"[INFO] Starting ICMP ping scan of {len(host_strings)} hosts...\n",
        )

        start_time = monotonic()
        # Use at most configured worker count, but not more than number of hosts.
        max_workers = min(self.options.max_workers, len(host_strings))
        if max_workers <= 0:
            max_workers = 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ip = {
                executor.submit(self._ping_once, ip_str): ip_str
                for ip_str in host_strings
            }

            for future in as_completed(future_to_ip):
                ip_str = future_to_ip[future]
                try:
                    is_alive = future.result()
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(
                        f"[DEBUG] Ping to {ip_str} failed with {exc!r}\n",
                    )
                    continue

                if is_alive:
                    live_hosts.append(ip_str)

        duration = monotonic() - start_time
        sys.stderr.write(
            f"[INFO] ICMP ping scan completed in {duration:.2f}s, "
            f"found {len(live_hosts)} live hosts.\n",
        )

        return live_hosts

    def _ping_once(self, ip: str) -> bool:
        """
        Perform a single ping to the given IP using the system 'ping' command.

        Platform-specific behavior:
        - Windows:
            ping -n 1 -w <timeout_ms> <ip>
        - Linux/macOS/BSD:
            ping -c 1 -W <timeout_s> <ip>

        Args:
            ip:
                Target IPv4 address as a string.

        Returns:
            True if ping succeeded (host responded), otherwise False.
        """
        system = platform.system().lower()

        if system == "windows":
            timeout_ms = max(1, int(self.options.icmp_timeout * 1000))
            cmd = [
                "ping",
                "-n",
                "1",
                "-w",
                str(timeout_ms),
                ip,
            ]
        else:
            # For UNIX-like systems:
            #   - '-c 1' sends a single ICMP echo request.
            #   - '-W <seconds>' or '-W <milliseconds>' acts as timeout.
            timeout_s = max(1, int(self.options.icmp_timeout))
            cmd = [
                "ping",
                "-c",
                "1",
                "-W",
                str(timeout_s),
                ip,
            ]

        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=self.options.icmp_timeout + 1.0,
            )
            return completed.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    # -------------------------------------------------------------------------
    # ARP table parsing
    # -------------------------------------------------------------------------

    def _read_arp_table(self) -> Dict[str, str]:
        """
        Read and parse the system ARP table, returning IP -> MAC mappings.

        Platforms:
        - Linux:  prefer 'ip neigh show', fallback to 'arp -n'.
        - macOS/BSD: 'arp -a'.
        - Windows:   'arp -a'.

        Parsing strategy:
        - Use regular expressions to extract IPv4 and MAC addresses from each
          line in the output, regardless of exact tool formatting.

        Returns:
            Dictionary mapping IPv4 address string to MAC address string.
        """
        ip_mac_map: Dict[str, str] = {}
        system = platform.system().lower()

        ip_regex = r"(?:\d{1,3}\.){3}\d{1,3}"
        mac_regex = r"(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}"
        ip_pattern = re.compile(ip_regex)
        mac_pattern = re.compile(mac_regex)

        def parse_output(output: str) -> None:
            for line in output.splitlines():
                ip_match = ip_pattern.search(line)
                mac_match = mac_pattern.search(line)
                if not (ip_match and mac_match):
                    continue
                ip_str = ip_match.group(0)
                mac_str = mac_match.group(0).lower()
                ip_mac_map[ip_str] = mac_str

        # Prefer 'ip neigh show' on Linux/macOS/BSD when available.
        if system in ("linux", "darwin", "freebsd"):
            ip_exe = shutil.which("ip")
            if ip_exe:
                try:
                    output = subprocess.check_output(
                        [ip_exe, "neigh", "show"],
                        text=True,
                        encoding="utf-8",
                        errors="ignore",
                    )
                    parse_output(output)
                    if ip_mac_map:
                        return ip_mac_map
                except (OSError, subprocess.SubprocessError):
                    # Fall through to other methods.
                    pass

        # Fallback to 'arp -a' or 'arp -n' depending on platform.
        arp_exe = shutil.which("arp")
        if not arp_exe:
            return ip_mac_map

        arp_cmd = [arp_exe]
        if system == "linux":
            arp_cmd.append("-n")
        else:
            arp_cmd.append("-a")

        try:
            output = subprocess.check_output(
                arp_cmd,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            parse_output(output)
        except (OSError, subprocess.SubprocessError):
            # If all methods fail, we simply return what we have (possibly empty).
            return ip_mac_map

        return ip_mac_map

    # -------------------------------------------------------------------------
    # Result aggregation and enrichment
    # -------------------------------------------------------------------------

    def _build_device_list_from_results(
        self,
        ip_mac_map: Dict[str, str],
        primary_ip: str,
    ) -> List[DetectedDevice]:
        """
        Convert raw IP->MAC mapping into a structured and enriched device list.

        Enrichment steps:
        - Mark which entry corresponds to the current machine (primary_ip).
        - Perform reverse-DNS hostname lookup for each IP when enabled.
        - Normalize missing MAC addresses to None.

        Args:
            ip_mac_map:
                Dictionary mapping IP string to MAC string (possibly empty
                string when MAC is unknown).
            primary_ip:
                The primary local IPv4 address (string) for the current host.

        Returns:
            List of DetectedDevice instances.
        """
        devices: List[DetectedDevice] = []

        # Ensure the primary host is included even if it was not discovered
        # via ARP or ping (e.g., if ARP table is incomplete).
        all_ips = set(ip_mac_map.keys())
        all_ips.add(primary_ip)

        for ip_str in all_ips:
            raw_mac = ip_mac_map.get(ip_str, "")
            mac_str = raw_mac.lower() if raw_mac else None
            is_self = (ip_str == primary_ip)
            hostname = (
                self._reverse_dns_lookup(ip_str)
                if self.options.enable_reverse_dns
                else None
            )
            devices.append(
                DetectedDevice(
                    ip=ip_str,
                    mac=mac_str,
                    hostname=hostname,
                    is_self=is_self,
                ),
            )

        return devices

    def _reverse_dns_lookup(self, ip: str) -> Optional[str]:
        """
        Attempt a reverse-DNS lookup to obtain a hostname for the given IP.

        Notes:
        - This may rely on local DNS, router-provided names, or mDNS.
        - Many IPs will not have a meaningful reverse mapping; errors are
          quietly ignored to keep the scan robust.

        Args:
            ip:
                IPv4 address as a string.

        Returns:
            Hostname string if available, else None.
        """
        try:
            host, _aliaslist, _ipaddrlist = socket.gethostbyaddr(ip)
            return host
        except (socket.herror, socket.gaierror, OSError):
            return None


def _format_devices_table(devices: List[DetectedDevice]) -> str:
    """
    Format the detected devices into a human-readable table.

    The table has four columns:
    - IP Address
    - MAC Address
    - Self
    - Hostname

    Args:
        devices:
            List of DetectedDevice objects to display.

    Returns:
        A multi-line string containing the formatted table.
    """
    headers = ["IP Address", "MAC Address", "Self", "Hostname"]

    rows: List[List[str]] = []
    for dev in devices:
        ip_str = dev.ip
        mac_str = dev.mac or "-"
        self_str = "Yes" if dev.is_self else ""
        host_str = dev.hostname or "-"
        rows.append([ip_str, mac_str, self_str, host_str])

    if not rows:
        rows.append(["-", "-", "-", "-"])

    # Compute per-column width based on header and all rows.
    columns = list(zip(*([headers] + rows)))
    col_widths = [
        max(len(str(item)) for item in col)
        for col in columns
    ]

    def format_row(row: List[str]) -> str:
        return "  ".join(
            str(cell).ljust(width) for cell, width in zip(row, col_widths)
        )

    lines: List[str] = []
    lines.append(format_row(headers))
    lines.append(
        "  ".join("-" * width for width in col_widths),
    )
    for row in rows:
        lines.append(format_row(row))

    return "\n".join(lines)


def _parse_args(argv: Optional[List[str]] = None) -> ScannerOptions:
    """
    Parse command-line arguments and build a ScannerOptions instance.

    Supported CLI options:
        --method {auto,arp,ping}
            Preferred scanning method. 'auto' tries ARP first (when Scapy is
            available), then falls back to ping. 'arp' forces only ARP (and
            falls back to ping if ARP fails). 'ping' disables ARP attempts.
        --icmp-timeout SECONDS
            Per-host ICMP timeout (float). Default: 0.75 seconds.
        --arp-timeout SECONDS
            ARP request timeout used by Scapy. Default: 2.0 seconds.
        --max-workers N
            Maximum number of concurrent ping workers. Default: 128.
        --no-rdns
            Disable reverse-DNS resolution (faster scans).
        --max-hosts N
            Safety limit for network size before confirmation is required.
            Default: 4096.
        --network CIDR
            Override auto-detected network with explicit CIDR, e.g.,
            "192.168.1.0/24".
    """
    parser = argparse.ArgumentParser(
        description="Discover devices on your local Wi-Fi / LAN.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "arp", "ping"],
        default="auto",
        help=(
            "Scan method: 'auto' (default) tries ARP then ping; "
            "'arp' prefers ARP; 'ping' uses ICMP only."
        ),
    )
    parser.add_argument(
        "--icmp-timeout",
        type=float,
        default=0.75,
        help="ICMP (ping) timeout per host in seconds (default: 0.75).",
    )
    parser.add_argument(
        "--arp-timeout",
        type=float,
        default=2.0,
        help="ARP timeout in seconds when using Scapy (default: 2.0).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=128,
        help="Maximum concurrent ping workers (default: 128).",
    )
    parser.add_argument(
        "--no-rdns",
        action="store_true",
        help="Disable reverse-DNS lookups for faster scans.",
    )
    parser.add_argument(
        "--max-hosts",
        type=int,
        default=4096,
        help=(
            "Safety limit on number of hosts in network before confirmation "
            "is required (default: 4096)."
        ),
    )
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        help=(
            "Optional explicit IPv4 network to scan, e.g. '192.168.1.0/24'. "
            "If provided, it must contain your primary IP."
        ),
    )

    args = parser.parse_args(argv)

    prefer_arp: bool
    if args.method == "ping":
        prefer_arp = False
    else:
        prefer_arp = True

    # If user explicitly chose "arp", we still allow ping fallback, but ARP
    # will be attempted first. For "auto" the behavior is the same by design.
    options = ScannerOptions(
        icmp_timeout=args.icmp_timeout,
        arp_timeout=args.arp_timeout,
        max_workers=args.max_workers,
        prefer_arp=prefer_arp,
        enable_reverse_dns=not args.no_rdns,
        max_hosts=args.max_hosts,
        override_network=args.network,
    )
    return options


def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for direct execution.

    Steps:
    - Parse command-line arguments into ScannerOptions.
    - Instantiate LocalNetworkScanner.
    - Execute scan().
    - Format and print the resulting table.
    """
    options = _parse_args(argv)
    scanner = LocalNetworkScanner(options=options)

    try:
        devices = scanner.scan()
    except KeyboardInterrupt:
        print("\n[INFO] Scan interrupted by user.", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[ERROR] Unexpected failure during scan: {exc!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    table = _format_devices_table(devices)
    print(table)


if __name__ == "__main__":
    main()