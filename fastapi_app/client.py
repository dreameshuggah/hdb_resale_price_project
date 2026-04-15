"""
client.py — Python client for the HDB Resale Price Prediction API.

Usage
-----
    # Basic prediction (uses built-in sample payload):
    python client.py

    # Fully custom prediction:
    python client.py \\
        --url http://localhost:8000 \\
        --flat-type "4 ROOM" \\
        --region "CENTRAL REGION" \\
        --town "TAMPINES" \\
        --storey "10 TO 12" \\
        --flat-model "Model A" \\
        --distance 14500 \\
        --area 93 \\
        --lease 65 \\
        --index 182.3

    # Explore valid categorical options:
    python client.py --metadata

    # Pretty-print help:
    python client.py --help
"""

import argparse
import json
import sys
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Default server URL
# ---------------------------------------------------------------------------
DEFAULT_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Sample payload (mirrors a typical Tampines 4-Room transaction)
# ---------------------------------------------------------------------------
SAMPLE_PAYLOAD: dict[str, Any] = {
    "flat_type": "4 ROOM",
    "region_ura": "EAST REGION",
    "town": "TAMPINES",
    "storey_range": "10 TO 12",
    "flat_model": "Model A",
    "distance_to_cbd": 14500.0,
    "floor_area_sqm": 93.0,
    "remaining_lease_years": 65.0,
    "resale_price_index": 182.3,
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _divider(char: str = "─", width: int = 60) -> str:
    return char * width


def _print_header(title: str) -> None:
    print()
    print(_divider("═"))
    print(f"  {title}")
    print(_divider("═"))


def _print_section(title: str) -> None:
    print()
    print(_divider())
    print(f"  {title}")
    print(_divider())


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def health_check(base_url: str) -> dict:
    """GET /  — check that the server is reachable."""
    response = requests.get(f"{base_url}/", timeout=10)
    response.raise_for_status()
    return response.json()


def get_metadata(base_url: str) -> dict:
    """GET /metadata  — retrieve valid categorical options."""
    response = requests.get(f"{base_url}/metadata", timeout=10)
    response.raise_for_status()
    return response.json()


def predict(base_url: str, payload: dict[str, Any]) -> dict:
    """POST /predict  — obtain a price prediction."""
    response = requests.post(
        f"{base_url}/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    # Surface validation / server errors clearly
    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        print(f"\n[ERROR] HTTP {response.status_code}: {_pretty_json(detail)}")
        sys.exit(1)
    return response.json()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="client.py",
        description="HDB Resale Price Prediction — FastAPI client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Server
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        metavar="URL",
        help=f"Base URL of the API server (default: {DEFAULT_URL})",
    )

    # Operational modes
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--metadata",
        action="store_true",
        help="Fetch and display valid categorical options, then exit.",
    )
    mode.add_argument(
        "--health",
        action="store_true",
        help="Perform a health check only, then exit.",
    )

    # Feature inputs (all optional — fall back to SAMPLE_PAYLOAD if omitted)
    feat = parser.add_argument_group("Feature inputs (all optional; defaults use sample payload)")
    feat.add_argument("--flat-type",  metavar="STR",   help="e.g. '4 ROOM'")
    feat.add_argument("--region",     metavar="STR",   help="e.g. 'EAST REGION'")
    feat.add_argument("--town",       metavar="STR",   help="e.g. 'TAMPINES'")
    feat.add_argument("--storey",     metavar="STR",   help="e.g. '10 TO 12'")
    feat.add_argument("--flat-model", metavar="STR",   help="e.g. 'Model A'")
    feat.add_argument("--distance",   metavar="FLOAT", type=float, help="Distance to CBD in metres")
    feat.add_argument("--area",       metavar="FLOAT", type=float, help="Floor area in sqm")
    feat.add_argument("--lease",      metavar="FLOAT", type=float, help="Remaining lease in years")
    feat.add_argument("--index",      metavar="FLOAT", type=float, help="HDB Resale Price Index value")

    return parser


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI args over the sample payload."""
    payload = SAMPLE_PAYLOAD.copy()
    mapping = {
        "flat_type":             args.flat_type,
        "region_ura":            args.region,
        "town":                  args.town,
        "storey_range":          args.storey,
        "flat_model":            args.flat_model,
        "distance_to_cbd":       args.distance,
        "floor_area_sqm":        args.area,
        "remaining_lease_years": args.lease,
        "resale_price_index":    args.index,
    }
    for key, value in mapping.items():
        if value is not None:
            payload[key] = value
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    # ── Health check ────────────────────────────────────────────────────────
    _print_header("HDB Resale Price Prediction — API Client")

    print("\n[1] Health Check …")
    try:
        health = health_check(base_url)
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Cannot reach the server at {base_url}.")
        print("        Make sure the server is running:")
        print("        uvicorn server:app --reload --host 0.0.0.0 --port 8000")
        sys.exit(1)

    print(f"    Status  : {health.get('status', 'unknown').upper()}")
    print(f"    Model   : {health.get('model', '—')}")
    print(f"    Docs    : {base_url}{health.get('docs', '/docs')}")

    if args.health:
        return  # Health-check-only mode

    # ── Metadata ─────────────────────────────────────────────────────────────
    if args.metadata:
        _print_section("Valid Categorical Options (GET /metadata)")
        meta = get_metadata(base_url)
        for field, values in meta.items():
            print(f"\n  {field}:")
            for v in values:
                print(f"    • {v}")
        return

    # ── Prediction ────────────────────────────────────────────────────────────
    payload = build_payload(args)

    _print_section("Prediction Request (POST /predict)")
    print("\n  Input payload:")
    for k, v in payload.items():
        print(f"    {k:<26} : {v}")

    print("\n[2] Sending prediction request …")
    result = predict(base_url, payload)

    _print_section("Prediction Result")
    price = result["predicted_resale_price"]
    print(f"\n  ┌{'─'*40}┐")
    print(f"  │  Predicted Resale Price : SGD {price:>12,.2f}  │")
    print(f"  └{'─'*40}┘")

    print(f"\n  Model file used : {result['model_file']}")

    print("\n  Encoded feature vector passed to model:")
    for k, v in result["input_features"].items():
        print(f"    {k:<26} : {v}")

    print()


if __name__ == "__main__":
    main()
