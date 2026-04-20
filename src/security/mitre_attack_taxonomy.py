"""MITRE ATT&CK technique taxonomy and heuristic classifier.

Pure stdlib implementation inspired by Anthropic-Cybersecurity-Skills MITRE
ATT&CK mapping domain. Provides a curated catalog of common enterprise
techniques, lookup utilities, and a keyword-driven classifier that tags
free-text threat descriptions with likely technique IDs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


TACTIC_ORDER: List[str] = [
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]


ATTACK_TECHNIQUES: Dict[str, Dict] = {
    # --- Initial Access ---
    "T1566": {
        "name": "Phishing",
        "tactic": ["initial-access"],
        "description": "Adversaries send phishing messages to gain access to victim systems.",
        "keywords": ["phishing", "phish", "spear-phish", "spearphishing", "malicious email"],
    },
    "T1566.001": {
        "name": "Spearphishing Attachment",
        "tactic": ["initial-access"],
        "description": "Phishing with a malicious file attachment.",
        "keywords": ["malicious attachment", "phishing attachment", "weaponized document", "malicious doc"],
    },
    "T1566.002": {
        "name": "Spearphishing Link",
        "tactic": ["initial-access"],
        "description": "Phishing using a malicious link.",
        "keywords": ["phishing link", "malicious link", "credential harvesting url"],
    },
    "T1190": {
        "name": "Exploit Public-Facing Application",
        "tactic": ["initial-access"],
        "description": "Exploiting a weakness in an Internet-facing application.",
        "keywords": ["exploit public", "web exploit", "rce web", "internet-facing vulnerability", "sql injection"],
    },
    "T1078": {
        "name": "Valid Accounts",
        "tactic": ["initial-access", "persistence", "privilege-escalation", "defense-evasion"],
        "description": "Use of compromised legitimate credentials.",
        "keywords": ["valid account", "compromised credentials", "stolen credentials", "legitimate account"],
    },
    "T1133": {
        "name": "External Remote Services",
        "tactic": ["initial-access", "persistence"],
        "description": "Leveraging external-facing remote services like VPN or RDP.",
        "keywords": ["vpn", "rdp external", "remote service", "citrix"],
    },
    "T1195": {
        "name": "Supply Chain Compromise",
        "tactic": ["initial-access"],
        "description": "Compromise via manipulated supply chain dependency.",
        "keywords": ["supply chain", "dependency confusion", "compromised package", "solarwinds"],
    },
    "T1199": {
        "name": "Trusted Relationship",
        "tactic": ["initial-access"],
        "description": "Abuse a trusted third-party relationship.",
        "keywords": ["trusted relationship", "msp compromise", "third-party access"],
    },
    "T1200": {
        "name": "Hardware Additions",
        "tactic": ["initial-access"],
        "description": "Introducing malicious hardware into a target environment.",
        "keywords": ["malicious usb", "rogue device", "hardware implant"],
    },

    # --- Execution ---
    "T1059": {
        "name": "Command and Scripting Interpreter",
        "tactic": ["execution"],
        "description": "Abuse of command/script interpreters to execute commands.",
        "keywords": ["command interpreter", "scripting interpreter", "shell script execution"],
    },
    "T1059.001": {
        "name": "PowerShell",
        "tactic": ["execution"],
        "description": "Use PowerShell to execute commands or scripts.",
        "keywords": ["powershell", "pwsh", "posh", "ps1"],
    },
    "T1059.003": {
        "name": "Windows Command Shell",
        "tactic": ["execution"],
        "description": "Use the Windows command shell (cmd.exe) to execute commands.",
        "keywords": ["cmd.exe", "windows command shell", "command prompt"],
    },
    "T1059.004": {
        "name": "Unix Shell",
        "tactic": ["execution"],
        "description": "Use Unix shells for execution.",
        "keywords": ["bash", "unix shell", "sh script", "zsh"],
    },
    "T1059.005": {
        "name": "Visual Basic",
        "tactic": ["execution"],
        "description": "Use Visual Basic / VBScript to execute code.",
        "keywords": ["vbscript", "visual basic", "vba macro"],
    },
    "T1059.006": {
        "name": "Python",
        "tactic": ["execution"],
        "description": "Use Python scripts for execution.",
        "keywords": ["python script", "python payload", "py2exe"],
    },
    "T1204": {
        "name": "User Execution",
        "tactic": ["execution"],
        "description": "Rely on user to open/execute a malicious file or link.",
        "keywords": ["user execution", "user opens", "social engineering execution"],
    },
    "T1053": {
        "name": "Scheduled Task/Job",
        "tactic": ["execution", "persistence", "privilege-escalation"],
        "description": "Abuse scheduled task/job facilities for execution and persistence.",
        "keywords": ["scheduled task", "cron job", "at job", "task scheduler"],
    },
    "T1053.005": {
        "name": "Scheduled Task",
        "tactic": ["execution", "persistence", "privilege-escalation"],
        "description": "Windows Task Scheduler used for execution/persistence.",
        "keywords": ["schtasks", "windows scheduled task"],
    },
    "T1569": {
        "name": "System Services",
        "tactic": ["execution"],
        "description": "Abuse system services for execution.",
        "keywords": ["system service", "service execution", "psexec"],
    },
    "T1047": {
        "name": "Windows Management Instrumentation",
        "tactic": ["execution"],
        "description": "Use WMI for execution.",
        "keywords": ["wmi", "wmic", "windows management instrumentation"],
    },

    # --- Persistence ---
    "T1547": {
        "name": "Boot or Logon Autostart Execution",
        "tactic": ["persistence", "privilege-escalation"],
        "description": "Configure system to auto-execute during boot/logon.",
        "keywords": ["autostart", "registry run key", "startup folder", "autorun"],
    },
    "T1543": {
        "name": "Create or Modify System Process",
        "tactic": ["persistence", "privilege-escalation"],
        "description": "Create or modify system-level processes for persistence.",
        "keywords": ["new service", "modify service", "systemd service", "launchd"],
    },
    "T1136": {
        "name": "Create Account",
        "tactic": ["persistence"],
        "description": "Create an account for persistent access.",
        "keywords": ["create account", "new local user", "add user backdoor"],
    },
    "T1098": {
        "name": "Account Manipulation",
        "tactic": ["persistence", "privilege-escalation"],
        "description": "Manipulate account permissions to maintain access.",
        "keywords": ["account manipulation", "add ssh key", "modify permissions"],
    },
    "T1505": {
        "name": "Server Software Component",
        "tactic": ["persistence"],
        "description": "Abuse server software components for persistence.",
        "keywords": ["web shell", "iis module", "sql stored procedure"],
    },
    "T1505.003": {
        "name": "Web Shell",
        "tactic": ["persistence"],
        "description": "Install a web shell on a compromised web server.",
        "keywords": ["webshell", "web shell", "china chopper", "aspx shell"],
    },
    "T1546": {
        "name": "Event Triggered Execution",
        "tactic": ["persistence", "privilege-escalation"],
        "description": "Execute code in response to specific events.",
        "keywords": ["wmi subscription", "event trigger", "accessibility features"],
    },

    # --- Privilege Escalation ---
    "T1068": {
        "name": "Exploitation for Privilege Escalation",
        "tactic": ["privilege-escalation"],
        "description": "Exploit a software vulnerability to escalate privileges.",
        "keywords": ["privilege escalation exploit", "kernel exploit", "local exploit"],
    },
    "T1548": {
        "name": "Abuse Elevation Control Mechanism",
        "tactic": ["privilege-escalation", "defense-evasion"],
        "description": "Abuse elevation mechanisms (UAC, sudo) to bypass controls.",
        "keywords": ["uac bypass", "sudo abuse", "setuid", "elevation bypass"],
    },
    "T1134": {
        "name": "Access Token Manipulation",
        "tactic": ["privilege-escalation", "defense-evasion"],
        "description": "Manipulate access tokens to run under a different user.",
        "keywords": ["token manipulation", "token impersonation", "access token"],
    },

    # --- Defense Evasion ---
    "T1027": {
        "name": "Obfuscated Files or Information",
        "tactic": ["defense-evasion"],
        "description": "Obfuscate payloads to evade detection.",
        "keywords": ["obfuscation", "obfuscated", "encoded payload", "base64 payload"],
    },
    "T1070": {
        "name": "Indicator Removal",
        "tactic": ["defense-evasion"],
        "description": "Remove artifacts/indicators from the system.",
        "keywords": ["clear logs", "delete logs", "indicator removal", "wevtutil cl"],
    },
    "T1070.004": {
        "name": "File Deletion",
        "tactic": ["defense-evasion"],
        "description": "Delete files to cover tracks.",
        "keywords": ["file deletion", "wipe artifact", "delete payload"],
    },
    "T1112": {
        "name": "Modify Registry",
        "tactic": ["defense-evasion"],
        "description": "Modify registry to hide configuration or aid evasion.",
        "keywords": ["modify registry", "registry edit", "reg add"],
    },
    "T1055": {
        "name": "Process Injection",
        "tactic": ["defense-evasion", "privilege-escalation"],
        "description": "Inject code into another process to evade defenses.",
        "keywords": ["process injection", "dll injection", "process hollowing", "reflective loading"],
    },
    "T1036": {
        "name": "Masquerading",
        "tactic": ["defense-evasion"],
        "description": "Masquerade malicious artifacts as legitimate ones.",
        "keywords": ["masquerading", "rename binary", "svchost impersonation"],
    },
    "T1218": {
        "name": "System Binary Proxy Execution",
        "tactic": ["defense-evasion"],
        "description": "Use trusted system binaries to proxy malicious execution.",
        "keywords": ["lolbin", "rundll32", "regsvr32", "mshta", "signed binary proxy"],
    },
    "T1562": {
        "name": "Impair Defenses",
        "tactic": ["defense-evasion"],
        "description": "Disable or tamper with security controls.",
        "keywords": ["disable antivirus", "disable defender", "impair defenses", "stop edr"],
    },

    # --- Credential Access ---
    "T1003": {
        "name": "OS Credential Dumping",
        "tactic": ["credential-access"],
        "description": "Dump credentials from the operating system.",
        "keywords": ["credential dumping", "lsass dump", "mimikatz", "sam dump", "ntds.dit"],
    },
    "T1003.001": {
        "name": "LSASS Memory",
        "tactic": ["credential-access"],
        "description": "Dump LSASS process memory for credentials.",
        "keywords": ["lsass", "lsass memory", "procdump lsass"],
    },
    "T1110": {
        "name": "Brute Force",
        "tactic": ["credential-access"],
        "description": "Systematically guess passwords/keys.",
        "keywords": ["brute force", "password spray", "credential stuffing", "password guessing"],
    },
    "T1552": {
        "name": "Unsecured Credentials",
        "tactic": ["credential-access"],
        "description": "Find credentials stored insecurely on systems.",
        "keywords": ["unsecured credentials", "hardcoded password", "credentials in file", "plaintext password"],
    },
    "T1555": {
        "name": "Credentials from Password Stores",
        "tactic": ["credential-access"],
        "description": "Extract credentials from browser or OS password stores.",
        "keywords": ["browser password", "keychain dump", "password store", "credential manager"],
    },
    "T1558": {
        "name": "Steal or Forge Kerberos Tickets",
        "tactic": ["credential-access"],
        "description": "Abuse Kerberos authentication tickets.",
        "keywords": ["kerberoasting", "golden ticket", "silver ticket", "kerberos abuse"],
    },
    "T1557": {
        "name": "Adversary-in-the-Middle",
        "tactic": ["credential-access", "collection"],
        "description": "Position between two devices to intercept/relay traffic.",
        "keywords": ["adversary in the middle", "mitm", "arp spoofing", "llmnr poisoning"],
    },

    # --- Discovery ---
    "T1083": {
        "name": "File and Directory Discovery",
        "tactic": ["discovery"],
        "description": "Enumerate files and directories.",
        "keywords": ["file discovery", "directory listing", "dir /s", "find files"],
    },
    "T1082": {
        "name": "System Information Discovery",
        "tactic": ["discovery"],
        "description": "Gather detailed information about the operating system.",
        "keywords": ["systeminfo", "uname", "os fingerprint", "system information"],
    },
    "T1057": {
        "name": "Process Discovery",
        "tactic": ["discovery"],
        "description": "Enumerate running processes.",
        "keywords": ["tasklist", "ps -ef", "process enumeration", "process discovery"],
    },
    "T1018": {
        "name": "Remote System Discovery",
        "tactic": ["discovery"],
        "description": "Identify other systems on the network.",
        "keywords": ["remote system discovery", "network enumeration", "net view", "host discovery"],
    },
    "T1087": {
        "name": "Account Discovery",
        "tactic": ["discovery"],
        "description": "Enumerate local or domain accounts.",
        "keywords": ["account discovery", "net user", "whoami", "enumerate accounts"],
    },
    "T1016": {
        "name": "System Network Configuration Discovery",
        "tactic": ["discovery"],
        "description": "Look up network configuration and settings.",
        "keywords": ["ipconfig", "ifconfig", "network configuration", "route print"],
    },
    "T1046": {
        "name": "Network Service Discovery",
        "tactic": ["discovery"],
        "description": "Scan network for listening services.",
        "keywords": ["port scan", "nmap", "service discovery", "network scan"],
    },

    # --- Lateral Movement ---
    "T1021": {
        "name": "Remote Services",
        "tactic": ["lateral-movement"],
        "description": "Use valid accounts to log into remote services.",
        "keywords": ["remote services", "lateral movement", "remote login"],
    },
    "T1021.001": {
        "name": "Remote Desktop Protocol",
        "tactic": ["lateral-movement"],
        "description": "Use RDP to move laterally.",
        "keywords": ["rdp", "remote desktop", "mstsc"],
    },
    "T1021.002": {
        "name": "SMB/Windows Admin Shares",
        "tactic": ["lateral-movement"],
        "description": "Use SMB admin shares for lateral movement.",
        "keywords": ["smb share", "admin$", "c$ share", "windows admin share"],
    },
    "T1021.004": {
        "name": "SSH",
        "tactic": ["lateral-movement"],
        "description": "Use SSH to move laterally between systems.",
        "keywords": ["ssh lateral", "ssh into", "ssh pivot"],
    },
    "T1570": {
        "name": "Lateral Tool Transfer",
        "tactic": ["lateral-movement"],
        "description": "Transfer tools between systems within the network.",
        "keywords": ["lateral tool transfer", "copy tool", "push binary"],
    },

    # --- Collection ---
    "T1005": {
        "name": "Data from Local System",
        "tactic": ["collection"],
        "description": "Collect data from the local system.",
        "keywords": ["collect local data", "local file collection", "harvest files"],
    },
    "T1039": {
        "name": "Data from Network Shared Drive",
        "tactic": ["collection"],
        "description": "Collect data from a network share.",
        "keywords": ["network share collection", "shared drive data"],
    },
    "T1056": {
        "name": "Input Capture",
        "tactic": ["collection", "credential-access"],
        "description": "Capture user input such as keystrokes.",
        "keywords": ["keylogger", "keystroke logging", "input capture"],
    },
    "T1113": {
        "name": "Screen Capture",
        "tactic": ["collection"],
        "description": "Capture screenshots of the victim's display.",
        "keywords": ["screenshot", "screen capture", "screengrab"],
    },
    "T1114": {
        "name": "Email Collection",
        "tactic": ["collection"],
        "description": "Collect email data.",
        "keywords": ["email collection", "mailbox dump", "pst file"],
    },
    "T1560": {
        "name": "Archive Collected Data",
        "tactic": ["collection"],
        "description": "Archive and compress data prior to exfiltration.",
        "keywords": ["archive data", "rar data", "zip data", "7zip staging"],
    },

    # --- Command and Control ---
    "T1071": {
        "name": "Application Layer Protocol",
        "tactic": ["command-and-control"],
        "description": "Use application-layer protocols for C2 communication.",
        "keywords": ["c2 channel", "http c2", "application layer protocol"],
    },
    "T1071.001": {
        "name": "Web Protocols",
        "tactic": ["command-and-control"],
        "description": "Use web protocols (HTTP/HTTPS) for C2.",
        "keywords": ["http beacon", "https c2", "web protocol c2"],
    },
    "T1071.004": {
        "name": "DNS",
        "tactic": ["command-and-control"],
        "description": "Use DNS for C2 communication.",
        "keywords": ["dns tunneling", "dns c2", "dns exfil"],
    },
    "T1105": {
        "name": "Ingress Tool Transfer",
        "tactic": ["command-and-control"],
        "description": "Transfer tools or files into the victim environment.",
        "keywords": ["ingress tool transfer", "download payload", "fetch second stage", "curl payload"],
    },
    "T1572": {
        "name": "Protocol Tunneling",
        "tactic": ["command-and-control"],
        "description": "Tunnel C2 traffic inside another protocol.",
        "keywords": ["protocol tunneling", "ssh tunnel", "icmp tunnel"],
    },
    "T1090": {
        "name": "Proxy",
        "tactic": ["command-and-control"],
        "description": "Use a proxy to obfuscate C2 traffic.",
        "keywords": ["proxy c2", "socks proxy", "tor c2"],
    },
    "T1573": {
        "name": "Encrypted Channel",
        "tactic": ["command-and-control"],
        "description": "Encrypt C2 traffic to evade inspection.",
        "keywords": ["encrypted channel", "tls c2", "custom crypto"],
    },

    # --- Exfiltration ---
    "T1041": {
        "name": "Exfiltration Over C2 Channel",
        "tactic": ["exfiltration"],
        "description": "Exfiltrate data over an existing C2 channel.",
        "keywords": ["exfiltration over c2", "data exfil c2"],
    },
    "T1048": {
        "name": "Exfiltration Over Alternative Protocol",
        "tactic": ["exfiltration"],
        "description": "Exfiltrate using an alternative protocol from C2.",
        "keywords": ["exfiltration alternative protocol", "ftp exfil", "dns exfiltration"],
    },
    "T1567": {
        "name": "Exfiltration Over Web Service",
        "tactic": ["exfiltration"],
        "description": "Exfiltrate via a legitimate web service.",
        "keywords": ["exfiltration web service", "upload to dropbox", "google drive exfil", "pastebin upload"],
    },
    "T1052": {
        "name": "Exfiltration Over Physical Medium",
        "tactic": ["exfiltration"],
        "description": "Exfiltrate via USB or other physical medium.",
        "keywords": ["usb exfil", "physical medium exfil"],
    },

    # --- Impact ---
    "T1486": {
        "name": "Data Encrypted for Impact",
        "tactic": ["impact"],
        "description": "Encrypt data to disrupt availability (ransomware).",
        "keywords": ["ransomware", "data encrypted", "file encryption impact", "lockbit", "encrypt for ransom"],
    },
    "T1490": {
        "name": "Inhibit System Recovery",
        "tactic": ["impact"],
        "description": "Delete backups or shadow copies to prevent recovery.",
        "keywords": ["delete shadow copies", "vssadmin delete", "inhibit recovery", "wbadmin delete"],
    },
    "T1489": {
        "name": "Service Stop",
        "tactic": ["impact"],
        "description": "Stop or disable services for impact.",
        "keywords": ["service stop", "stop service", "kill service"],
    },
    "T1485": {
        "name": "Data Destruction",
        "tactic": ["impact"],
        "description": "Destroy data on target systems.",
        "keywords": ["data destruction", "wipe data", "destroy files", "wiper malware"],
    },
    "T1498": {
        "name": "Network Denial of Service",
        "tactic": ["impact"],
        "description": "Perform network DoS to disrupt availability.",
        "keywords": ["ddos", "dos attack", "denial of service"],
    },
    "T1496": {
        "name": "Resource Hijacking",
        "tactic": ["impact"],
        "description": "Hijack compute resources (e.g. cryptomining).",
        "keywords": ["cryptomining", "cryptojacking", "resource hijacking", "coin miner"],
    },
    "T1491": {
        "name": "Defacement",
        "tactic": ["impact"],
        "description": "Deface or modify content for impact.",
        "keywords": ["defacement", "website deface", "modify homepage"],
    },
}


_TECHNIQUE_ID_PATTERN = re.compile(r"^T\d+(\.\d+)?$")
_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-\._/]*")


@dataclass
class TechniqueMatch:
    technique_id: str
    name: str
    tactic: List[str]
    confidence: float
    matched_keywords: List[str] = field(default_factory=list)


class MitreAttackClassifier:
    """Heuristic keyword-based MITRE ATT&CK technique classifier."""

    def __init__(self, custom_techniques: Optional[Dict[str, Dict]] = None) -> None:
        catalog: Dict[str, Dict] = {tid: dict(entry) for tid, entry in ATTACK_TECHNIQUES.items()}
        if custom_techniques:
            for tid, entry in custom_techniques.items():
                if not _TECHNIQUE_ID_PATTERN.match(tid):
                    raise ValueError(f"Invalid technique id: {tid}")
                catalog[tid] = dict(entry)
        self._catalog: Dict[str, Dict] = catalog
        # Precompute normalized keyword list per technique.
        self._keyword_index: Dict[str, List[str]] = {
            tid: [kw.lower() for kw in entry.get("keywords", [])]
            for tid, entry in self._catalog.items()
        }

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def lookup(self, technique_id: str) -> Dict:
        if technique_id not in self._catalog:
            raise KeyError(f"Unknown technique id: {technique_id}")
        return dict(self._catalog[technique_id])

    def by_tactic(self, tactic: str) -> List[str]:
        if tactic not in TACTIC_ORDER:
            raise ValueError(f"Unknown tactic: {tactic}")
        return sorted(
            tid
            for tid, entry in self._catalog.items()
            if tactic in entry.get("tactic", [])
        )

    def get_kill_chain(self, technique_ids: Iterable[str]) -> List[str]:
        def sort_key(tid: str):
            entry = self._catalog.get(tid)
            if not entry:
                return (len(TACTIC_ORDER), tid)
            tactics = entry.get("tactic", [])
            positions = [TACTIC_ORDER.index(t) for t in tactics if t in TACTIC_ORDER]
            pos = min(positions) if positions else len(TACTIC_ORDER)
            return (pos, tid)

        return sorted(technique_ids, key=sort_key)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def classify(self, text: str, top_k: int = 5) -> List[TechniqueMatch]:
        if not text or not text.strip():
            return []
        lowered = text.lower()

        matches: List[TechniqueMatch] = []
        for tid, keywords in self._keyword_index.items():
            if not keywords:
                continue
            hit_kws: List[str] = []
            for kw in keywords:
                # Use substring match; works for multiword keywords and
                # single-token keywords alike.
                if kw and kw in lowered:
                    hit_kws.append(kw)
            if not hit_kws:
                continue
            # Confidence: fraction of this technique's keywords seen, with
            # a small bonus for multiple distinct hits. Clamped to [0,1].
            total = len(keywords)
            frac = len(hit_kws) / total
            bonus = min(0.2, 0.05 * (len(hit_kws) - 1))
            confidence = min(1.0, frac + bonus)
            entry = self._catalog[tid]
            matches.append(
                TechniqueMatch(
                    technique_id=tid,
                    name=entry["name"],
                    tactic=list(entry.get("tactic", [])),
                    confidence=round(confidence, 4),
                    matched_keywords=sorted(hit_kws),
                )
            )

        # Sort by confidence desc, then deterministically by technique_id.
        matches.sort(key=lambda m: (-m.confidence, m.technique_id))
        if top_k is not None and top_k >= 0:
            matches = matches[:top_k]
        return matches


__all__ = [
    "TACTIC_ORDER",
    "ATTACK_TECHNIQUES",
    "TechniqueMatch",
    "MitreAttackClassifier",
]
