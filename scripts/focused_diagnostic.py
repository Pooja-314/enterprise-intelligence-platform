"""Focused Diagnostic Script: DNS, TCP, Read-Only Database, and LLM Tests.
Does NOT run migrations or modify the database.
Does NOT print credentials or secrets.
"""
import os
import re
import socket
import subprocess
import sys
import time
import urllib.parse

from typing import Any, Dict, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(override=True)

from sqlalchemy import create_engine, text
from app.db.session import get_migration_database_url, get_normalized_database_url


def scrub_error(msg: str) -> str:
    """Removes sensitive credentials from error messages."""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", str(msg))


def resolve_with_powershell(hostname: str, dns_server: Optional[str] = None) -> Dict[str, Any]:
    cmd = ["powershell", "-NoProfile", "-Command", f"Resolve-DnsName -Name '{hostname}'" + (f" -Server '{dns_server}'" if dns_server else "") + " -ErrorAction Stop | Select-Object -ExpandProperty IPAddress"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if res.returncode == 0:
            ips = [ip.strip() for ip in res.stdout.strip().split() if ip.strip()]
            return {"status": "PASS", "ips": ips}
        else:
            err = res.stderr.strip() or res.stdout.strip()
            return {"status": "FAIL", "error": err.split("\n")[0]}
    except Exception as exc:
        return {"status": "FAIL", "error": str(exc)}


def run_focused_diagnostic():
    print("=" * 70)
    print("ENTERPRISE INTELLIGENCE PLATFORM - FOCUSED DIAGNOSTIC AUDIT")
    print("=" * 70)

    # Extract hostnames safely
    db_url = os.getenv("DATABASE_URL", "")
    direct_url = os.getenv("DIRECT_URL", "")
    parsed_db = urllib.parse.urlparse(db_url)
    parsed_direct = urllib.parse.urlparse(direct_url)

    pooled_host = parsed_db.hostname or ""
    direct_host = parsed_direct.hostname or ""

    pooled_port = parsed_db.port or 5432
    direct_port = parsed_direct.port or 5432

    # --------------------------------------------------------------------------
    # 1. Neon DNS Diagnostics
    # --------------------------------------------------------------------------
    print("\n--- [1] NEON DNS RESOLUTION DIAGNOSTICS ---")
    print(f"Pooled Endpoint Hostname: {pooled_host}")
    print(f"Direct Endpoint Hostname: {direct_host}\n")

    dns_results = {}

    for label, host in [("Pooled Endpoint", pooled_host), ("Direct Endpoint", direct_host)]:
        print(f"Testing {label} ({host}):")
        if not host:
            print("  Status: NOT CONFIGURED")
            continue

        # 1a. System Resolver (socket.getaddrinfo)
        try:
            addr_info = socket.getaddrinfo(host, 5432, socket.AF_INET, socket.SOCK_STREAM)
            sys_ips = sorted(list({info[4][0] for info in addr_info}))
            print(f"  [System Resolver]: PASS -> {sys_ips}")
            dns_results[f"{label}_system"] = {"status": "PASS", "ips": sys_ips}
        except Exception as exc:
            print(f"  [System Resolver]: FAIL -> {exc}")
            dns_results[f"{label}_system"] = {"status": "FAIL", "error": str(exc)}

        # 1b. Public DNS 8.8.8.8
        res_8 = resolve_with_powershell(host, "8.8.8.8")
        if res_8["status"] == "PASS":
            print(f"  [Public DNS 8.8.8.8]: PASS -> {res_8['ips']}")
        else:
            print(f"  [Public DNS 8.8.8.8]: FAIL -> {res_8.get('error')}")
        dns_results[f"{label}_8.8.8.8"] = res_8

        # 1c. Public DNS 1.1.1.1
        res_1 = resolve_with_powershell(host, "1.1.1.1")
        if res_1["status"] == "PASS":
            print(f"  [Public DNS 1.1.1.1]: PASS -> {res_1['ips']}")
        else:
            print(f"  [Public DNS 1.1.1.1]: FAIL -> {res_1.get('error')}")
        dns_results[f"{label}_1.1.1.1"] = res_1

    # --------------------------------------------------------------------------
    # 2. PostgreSQL TCP & Read-Only Connectivity
    # --------------------------------------------------------------------------
    print("\n--- [2] POSTGRESQL TCP & READ-ONLY CONNECTIVITY ---")

    for label, host, port, url in [
        ("Neon Pooled Endpoint", pooled_host, pooled_port, get_normalized_database_url(db_url)),
        ("Neon Direct Endpoint", direct_host, direct_port, get_migration_database_url())
    ]:
        print(f"\nTesting {label} (Port {port}):")
        if not host:
            print("  Status: NOT CONFIGURED")
            continue

        # TCP Socket Test
        tcp_ok = False
        try:
            start_tcp = time.perf_counter()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((host, port))
            s.close()
            latency_tcp = (time.perf_counter() - start_tcp) * 1000
            print(f"  TCP Handshake (Port {port}): PASS (Latency: {latency_tcp:.2f}ms)")
            tcp_ok = True
        except Exception as exc:
            print(f"  TCP Handshake (Port {port}): FAIL ({exc})")

        # Read-Only SELECT version() test
        if tcp_ok:
            try:
                start_sql = time.perf_counter()
                engine = create_engine(url, connect_args={"connect_timeout": 6})
                with engine.connect() as conn:
                    version = conn.execute(text("SELECT version();")).scalar()
                latency_sql = (time.perf_counter() - start_sql) * 1000
                v_short = version.split(",")[0] if version else "PostgreSQL"
                print(f"  Read-Only SELECT version(): PASS (Latency: {latency_sql:.2f}ms)")
                print(f"  Server: {v_short}")
            except Exception as exc:
                err = scrub_error(str(exc))
                print(f"  Read-Only SELECT version(): FAIL ({err})")
        else:
            print("  Read-Only SELECT version(): NOT VERIFIED (TCP failed)")

    # --------------------------------------------------------------------------
    # 3. LLM Connectivity
    # --------------------------------------------------------------------------
    print("\n--- [3] LLM API CONNECTIVITY ---")

    # 3a. Groq
    groq_model = os.getenv("GROQ_MODEL", "")
    print(f"\n[Groq API Probe] (Model: {groq_model})")
    from app.llm.groq_provider import GroqProvider
    groq = GroqProvider()
    if not groq.is_configured():
        print("  Authentication: FAIL (GROQ_API_KEY missing or placeholder)")
        print("  Model Availability: NOT VERIFIED")
        print("  Response Status: NOT VERIFIED")
    else:
        print("  Authentication: PASS (Key configured)")
        try:
            start_g = time.perf_counter()
            resp = groq.generate(prompt="Reply with the exact word: ONLINE", max_tokens=5)
            lat_g = (time.perf_counter() - start_g) * 1000
            print(f"  Model Availability: PASS ({resp.model})")
            print(f"  Response Status: PASS (Latency: {lat_g:.2f}ms, Output: '{resp.content.strip()}')")
        except Exception as exc:
            err = scrub_error(str(exc))
            if "model_not_found" in err or "404" in err:
                print(f"  Model Availability: FAIL (Model '{groq_model}' not found on Groq API)")
            else:
                print(f"  Model Availability: FAIL ({err})")
            print(f"  Response Status: FAIL")

    # 3b. Gemini
    gemini_model = os.getenv("GEMINI_MODEL", "")
    print(f"\n[Gemini API Probe] (Model: {gemini_model})")
    from app.llm.gemini_provider import GeminiProvider
    gemini = GeminiProvider()
    if not gemini.is_configured():
        print("  Authentication: FAIL (GEMINI_API_KEY missing or placeholder)")
        print("  Model Availability: NOT VERIFIED")
        print("  Response Status: NOT VERIFIED")
    else:
        print("  Authentication: PASS (Key configured)")
        try:
            start_gem = time.perf_counter()
            resp = gemini.generate(prompt="Reply with the exact word: ONLINE", max_tokens=5)
            lat_gem = (time.perf_counter() - start_gem) * 1000
            print(f"  Model Availability: PASS ({resp.model})")
            print(f"  Response Status: PASS (Latency: {lat_gem:.2f}ms, Output: '{resp.content.strip()}')")
        except Exception as exc:
            err = scrub_error(str(exc))
            if "NOT_FOUND" in err or "404" in err:
                print(f"  Model Availability: FAIL (Model '{gemini_model}' not found on Google GenAI API)")
            else:
                print(f"  Model Availability: FAIL ({err})")
            print(f"  Response Status: FAIL")

    print("\n" + "=" * 70)
    print("FOCUSED DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_focused_diagnostic()
