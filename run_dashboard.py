"""Serve the React dashboard and saved experiment results on localhost."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from fairlens.dashboard_server import make_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workspace-dir", type=Path, default=ROOT / "workspace",
                        help="Local directory for uploaded datasets and isolated experiments")
    args = parser.parse_args()
    server = make_server(ROOT / "results", ROOT / "frontend" / "dist", args.port, args.workspace_dir)
    print(f"FairLens dashboard: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
