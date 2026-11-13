"""Smoke-test a deployed study service.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request


def request(url: str, path: str, *, payload: dict | None = None, token: str = "") -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{url.rstrip('/')}{path}", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a deployed study service.")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--code", required=True)
    parser.add_argument("--websocket", action="store_true")
    parser.add_argument("--day", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()

    health = request(args.url, "/health")
    login = request(args.url, "/api/login", payload={"code": args.code, "role": "study"})
    bootstrap = request(args.url, "/api/bootstrap", token=login["token"])
    result = {
        "health": health,
        "cohort": login["cohort"],
        "group": login["group_name"],
        "consent_version": bootstrap["consent"]["version"],
        "consent_approved": bootstrap["consent"]["approved"],
        "formal_ready": bootstrap["formal_ready"],
    }
    if args.websocket:
        import websocket

        if not bootstrap["consent_valid"]:
            raise RuntimeError(
                "WebSocket pairing requires a participant who has personally completed consent"
            )
        companion = request(
            args.url, "/api/login", payload={"code": args.code, "role": "companion"}
        )
        study_session = request(
            args.url, "/api/sessions/start", payload={"day": args.day}, token=login["token"]
        )
        companion_session = request(
            args.url,
            "/api/sessions/start",
            payload={"day": args.day},
            token=companion["token"],
        )
        for kind in ("network", "microphone", "reaction_latency", "browser"):
            request(
                args.url,
                "/api/quality",
                payload={
                    "session_id": study_session["id"],
                    "kind": kind,
                    "passed": True,
                    "metrics": {"smoke_test": True},
                },
                token=login["token"],
            )
        parsed = urllib.parse.urlsplit(args.url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_base = f"{ws_scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        study_ws = websocket.create_connection(
            f"{ws_base}/ws/pair?token={urllib.parse.quote(login['token'])}"
            f"&session_id={urllib.parse.quote(study_session['id'])}",
            timeout=15,
        )
        companion_ws = websocket.create_connection(
            f"{ws_base}/ws/pair?token={urllib.parse.quote(companion['token'])}"
            f"&session_id={urllib.parse.quote(companion_session['id'])}",
            timeout=15,
        )
        try:
            companion_ws.send(
                json.dumps({"type": "background_state", "action": "ready", "ready": True})
            )
            pair_status = {}
            for _ in range(8):
                candidate = json.loads(study_ws.recv())
                if candidate.get("type") == "pair_status" and candidate.get("companion_ready"):
                    pair_status = candidate
                    break
            activation = request(
                args.url,
                f"/api/sessions/{study_session['id']}/activate",
                payload={},
                token=login["token"],
            )
            result["pair_qualified"] = bool(
                pair_status.get("same_day_online")
                and pair_status.get("day") == args.day
                and activation.get("pair_qualified")
            )
            study_ws.send(json.dumps({"type": "heartbeat", "probe": "public-smoke"}))
            relayed = {}
            for _ in range(5):
                candidate = json.loads(companion_ws.recv())
                if candidate.get("type") == "heartbeat":
                    relayed = candidate
                    break
            result["websocket_relay"] = (
                relayed.get("type") == "heartbeat"
                and relayed.get("probe") == "public-smoke"
                and relayed.get("from") == "study"
            )
            result["websocket_observed"] = {
                key: relayed.get(key) for key in ("type", "probe", "from")
            }
        finally:
            study_ws.close()
            companion_ws.close()

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
