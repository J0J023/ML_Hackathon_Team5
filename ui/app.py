"""Evaluator UI runner for Laptop 2 (Client).

Usage:
    # 1. Connect to Laptop 1 running on the same network:
    python ui/app.py --server http://192.168.1.100:8000

    # 2. Or test locally if running both on the same laptop:
    python ui/app.py --server http://localhost:8000
"""

import argparse
import http.server
import os
import socketserver
import sys
import webbrowser

UI_DIR = os.path.dirname(os.path.abspath(__file__))


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def log_message(self, format, *args):
        # Clean logging
        sys.stdout.write(f"[UI CLIENT] {self.address_string()} - {format % args}\n")


def main():
    parser = argparse.ArgumentParser(description="PathSnap Evaluator Interface Runner (Laptop 2)")
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="URL of Laptop 1 PathSnap API (e.g. http://192.168.1.100:8000)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to serve the Evaluator UI on Laptop 2 (default: 3000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the default web browser",
    )
    args = parser.parse_args()

    # Inject server URL into index.html if needed or pass as URL param
    url = f"http://localhost:{args.port}/index.html"

    print("=" * 70)
    print("  PATHSNAP EVALUATOR INTERFACE (Laptop 2 Client)")
    print(f"  Target Server (Laptop 1) : {args.server}")
    print(f"  Local UI Endpoint        : {url}")
    print("=" * 70)
    print(f"Serving UI files from: {UI_DIR}")
    print("Press Ctrl+C to stop the UI server.\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    with socketserver.TCPServer(("", args.port), CustomHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Evaluator UI...")


if __name__ == "__main__":
    main()
