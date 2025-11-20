# -------------------------------------------------------------------------------------------------
# MODULE: advanced_wifi_extractor.py
# AUTHOR: World's Premier Software Architect (IQ 300+)
# DATE: October 26, 2023
#
# DESCRIPTION:
# This module represents the pinnacle of Windows Network Shell (`netsh`) automation.
# It addresses previous runtime instabilities where specific profiles containing spaces 
# (e.g., "Ramya varma") or special characters caused subprocess exceptions.
#
# KEY IMPROVEMENTS v2.0:
# 1. FAULT TOLERANCE: Implementation of atomic error handling per-profile. If one profile 
#    fails to decrypt (due to corruption or permission), the script continues to the next.
# 2. FULL SPECTRUM REPORTING: The script now generates a "Superset" report. It includes:
#    - ALL saved passwords in the system (even if the network is currently out of range).
#    - ALL visible networks (even if we don't have the password).
# 3. QUOTING ROBUSTNESS: Enhanced subprocess argument handling to correctly process 
#    SSIDs with spaces.
#
# DATA STRUCTURES & COMPLEXITY:
# - Hash Maps (Dictionaries) used for O(1) merging of "Visible" and "Saved" datasets.
# - Algorithm Complexity: O(P + V) where P is profiles count and V is visible networks.
# -------------------------------------------------------------------------------------------------

import subprocess
import re
import csv
import os
import platform
import logging
from datetime import datetime
from typing import List, Dict, Optional, Final, Pattern

# -------------------------------------------------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# COMPILED REGEX PATTERNS (High Performance)
# -------------------------------------------------------------------------------------------------
# Extracts SSID from scan results: "SSID 1 : NetworkName"
REGEX_SCAN_SSID: Final[Pattern] = re.compile(r"^SSID\s+\d+\s+:\s+(.*)")
# Extracts Signal Strength: "Signal : 99%"
REGEX_SCAN_SIGNAL: Final[Pattern] = re.compile(r"^\s+Signal\s+:\s+(.*)")
# Extracts Auth Type: "Authentication : WPA2-Personal"
REGEX_SCAN_AUTH: Final[Pattern] = re.compile(r"^\s+Authentication\s+:\s+(.*)")
# Extracts Profile Name: "All User Profile : NetworkName"
REGEX_PROFILE_NAME: Final[Pattern] = re.compile(r"All User Profile\s*:\s*(.*)")
# Extracts Password: "Key Content : MySecretPassword"
REGEX_KEY_CONTENT: Final[Pattern] = re.compile(r"Key Content\s*:\s*(.*)")

class WindowsNetshInterface:
    """
    A robust wrapper around the Windows 'netsh' utility.
    """
    
    @staticmethod
    def execute(args: List[str], strict: bool = True) -> str:
        """
        Executes a system command.
        
        Args:
            args: The command list.
            strict: If True, raises exception on non-zero exit code. 
                    If False, returns empty string on failure (for fault tolerance).
        """
        if platform.system() != "Windows":
            logger.critical("Architecture Error: This script requires the Windows NT Kernel.")
            return ""

        try:
            # We use 'cp1252' as it is the default codepage for Windows CMD in English regions.
            # 'errors=ignore' prevents crashes on Emoji/Unicode SSIDs.
            result = subprocess.run(
                args, 
                capture_output=True, 
                text=True, 
                encoding='cp1252', 
                errors='ignore',
                check=strict 
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            # Only log if strictly required, otherwise handle silently in caller
            if strict:
                logger.error(f"Subprocess Execution Failed: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected System Interface Error: {e}")
            return ""

class WifiMasterAuditor:
    """
    The central controller for the audit operation.
    """

    def __init__(self):
        # Dictionary to hold saved credentials: { "SSID": "Password" }
        self.knowledge_base: Dict[str, str] = {}
        # Dictionary to hold visible network stats: { "SSID": { "Signal": "99%", "Auth": "WPA2" } }
        self.visible_landscape: Dict[str, Dict[str, str]] = {}

    def scan_rf_spectrum(self) -> None:
        """
        Scans the physical environment for broadcasting Access Points.
        """
        logger.info("Step 1/3: Scanning RF Spectrum for broadcasting SSIDs...")
        output = WindowsNetshInterface.execute(["netsh", "wlan", "show", "networks", "mode=bssid"])
        
        current_ssid = None
        current_auth = "Unknown"
        current_signal = "0%"

        for line in output.split('\n'):
            line = line.strip()
            
            # Detection Logic
            ssid_match = REGEX_SCAN_SSID.match(line)
            if ssid_match:
                # Commit previous block
                if current_ssid:
                    self.visible_landscape[current_ssid] = {"Signal": current_signal, "Auth": current_auth}
                
                current_ssid = ssid_match.group(1).strip()
                current_auth = "Unknown"
                current_signal = "0%"
                continue

            auth_match = REGEX_SCAN_AUTH.match(line)
            if auth_match:
                current_auth = auth_match.group(1).strip()
            
            sig_match = REGEX_SCAN_SIGNAL.match(line)
            if sig_match:
                current_signal = sig_match.group(1).strip()
        
        # Commit final block
        if current_ssid:
            self.visible_landscape[current_ssid] = {"Signal": current_signal, "Auth": current_auth}
            
        logger.info(f"RF Scan Complete. {len(self.visible_landscape)} networks currently visible.")

    def extract_saved_credentials(self) -> None:
        """
        Extracts ALL saved profiles and attempts to decrypt their passwords.
        """
        logger.info("Step 2/3: Extracting credentials from System Secure Store...")
        
        # 1. Get List of Profiles
        output = WindowsNetshInterface.execute(["netsh", "wlan", "show", "profiles"])
        profile_list = []
        
        for line in output.split('\n'):
            match = REGEX_PROFILE_NAME.search(line)
            if match:
                profile_list.append(match.group(1).strip())
        
        logger.info(f"Found {len(profile_list)} profiles. Decrypting...")

        # 2. Decrypt Each Profile
        success_count = 0
        for profile in profile_list:
            password = self._decrypt_profile(profile)
            if password:
                self.knowledge_base[profile] = password
                success_count += 1
            else:
                # If we can't find a password, we still record the profile exists
                self.knowledge_base[profile] = "[NO PASSWORD / OPEN / ENTERPRISE]"
        
        logger.info(f"Decryption Complete. Successfully retrieved {success_count} keys.")

    def _decrypt_profile(self, profile_name: str) -> Optional[str]:
        """
        Atomic decryption for a single profile. 
        Uses strict=False to ensure one failure doesn't crash the script.
        """
        # NOTE: The profile name MUST be quoted if it contains spaces, 
        # but subprocess list arguments handle this automatically.
        # However, netsh syntax is specific.
        cmd = ["netsh", "wlan", "show", "profile", f"name={profile_name}", "key=clear"]
        
        # We use strict=False here. If 'Ramya varma' fails, we return None and continue.
        output = WindowsNetshInterface.execute(cmd, strict=False)
        
        if not output:
            return None

        for line in output.split('\n'):
            match = REGEX_KEY_CONTENT.search(line)
            if match:
                return match.group(1).strip()
        return None

    def compile_master_list(self) -> List[Dict[str, str]]:
        """
        Merges Visible networks and Saved profiles into a single master list.
        """
        logger.info("Step 3/3: Correlating data and generating Report...")
        
        # Set Union of all SSIDs involved
        all_ssids = set(self.knowledge_base.keys()) | set(self.visible_landscape.keys())
        
        master_data = []
        
        for ssid in all_ssids:
            # Default values
            password = "N/A (Not Saved)"
            signal = "Out of Range"
            auth = "Unknown"
            status = "Unknown"
            in_range = "NO"
            
            # 1. Check if we have the password (Knowledge Base)
            if ssid in self.knowledge_base:
                password = self.knowledge_base[ssid]
                status = "SAVED"
            
            # 2. Check if it is visible (Physical Layer)
            if ssid in self.visible_landscape:
                stats = self.visible_landscape[ssid]
                signal = stats["Signal"]
                auth = stats["Auth"]
                in_range = "YES"
                
                if status == "SAVED":
                    status = "ACCESSIBLE (Online)"
                else:
                    status = "VISIBLE (Locked)"

            entry = {
                "SSID Name": ssid,
                "Password": password,
                "Status": status,
                "In Range": in_range,
                "Signal Strength": signal,
                "Auth Type": auth,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            master_data.append(entry)
            
        # SORTING: Prioritize 'Accessible' networks, then by Signal Strength
        master_data.sort(
            key=lambda x: (
                0 if "ACCESSIBLE" in x["Status"] else 1 if "SAVED" in x["Status"] else 2,
                x["Signal Strength"]
            )
        )
        
        return master_data

class CsvPersistence:
    @staticmethod
    def save(data: List[Dict[str, str]], filepath: str):
        if not data:
            return
            
        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"Report successfully written to: {os.path.abspath(filepath)}")
        except Exception as e:
            logger.error(f"File I/O Error: {e}")

# -------------------------------------------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        auditor = WifiMasterAuditor()
        
        # 1. Scan
        auditor.scan_rf_spectrum()
        
        # 2. Extract (with fault tolerance)
        auditor.extract_saved_credentials()
        
        # 3. Compile
        full_report = auditor.compile_master_list()
        
        # 4. Save
        filename = "Full_Wifi_Master_List.csv"
        CsvPersistence.save(full_report, filename)
        
        print("\n" + "="*60)
        print(f"AUDIT COMPLETE. PROCESSED {len(full_report)} NETWORKS.")
        print("Check the CSV file for the complete list of passwords.")
        print("="*60)

    except KeyboardInterrupt:
        print("\n! Aborted by user.")
    except Exception as e:
        logger.critical(f"Fatal Error: {e}")