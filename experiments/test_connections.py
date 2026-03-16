"""
test_connections.py — Verify API connectivity for all three model servers.

Makes a minimal call to each server and reports:
  - Reachability
  - Accepted model name
  - Response latency
  - A short sample response

Usage:
    python experiments/test_connections.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, make_client

ROLES = ["debater_a", "debater_b", "judge"]
TEST_PROMPT = "Reply with one word: hello."


def test_role(role: str, config: dict):
    print(f"\n{'─'*50}")
    print(f"  Role    : {role}")
    try:
        client, model = make_client(role, config)
        print(f"  Model   : {model}")

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=10,
            temperature=0,
        )
        latency = time.perf_counter() - t0

        text = response.choices[0].message.content or ""
        print(f"  Status  : OK")
        print(f"  Latency : {latency:.2f}s")
        print(f"  Response: {text.strip()!r}")
        return True

    except Exception as e:
        print(f"  Status  : FAILED")
        print(f"  Error   : {e}")
        return False


def main():
    config = load_config()
    print("Testing API connections...\n")

    results = {}
    for role in ROLES:
        results[role] = test_role(role, config)

    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    all_ok = True
    for role, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {role:<12} : {status}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  All connections OK. Ready to run experiments.")
    else:
        print("  Some connections failed. Check your .env and server status.")
    print()


if __name__ == "__main__":
    main()
