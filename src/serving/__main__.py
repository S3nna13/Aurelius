"""Entry point for `python -m src.serving`."""
from __future__ import annotations

import argparse
import logging

from .api_server import create_server, make_mock_generate_fn

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Aurelius API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    args = parser.parse_args()

    generate_fn = make_mock_generate_fn()
    server = create_server(args.host, args.port, generate_fn)
    logger.info("Aurelius API server listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
