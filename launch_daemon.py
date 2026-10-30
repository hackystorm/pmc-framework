#!/usr/bin/env python3
import asyncio
import argparse
import sys
from pathlib import Path

# Add repo root to path so this runs from a checkout as well as an install.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The package name, not the source tree: importing through `src.` builds a
# second copy of every pmc module, and the daemon ends up holding classes that
# are not the ones the rest of the framework uses.
from pmc.runtime.frenesis_daemon import (
    FRENESISDaemon,
    create_daemon_config,
)


def main():
    parser = argparse.ArgumentParser(description="FRENESIS Autonomous Agent Daemon")
    parser.add_argument("--workspace", required=True, help="FRENESIS workspace path")
    parser.add_argument("--llama-model", required=True, help="Path to GGUF model")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server executable")
    parser.add_argument("--llama-port", type=int, default=8080, help="llama-server port")
    parser.add_argument("--llama-host", default="127.0.0.1", help="llama-server host")
    parser.add_argument("--llama-ctx-size", type=int, default=4096, help="Context size")
    parser.add_argument("--llama-threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--http-port", type=int, default=8081, help="Daemon HTTP port")
    parser.add_argument("--http-host", default="127.0.0.1", help="Daemon HTTP host")
    parser.add_argument("--ws-port", type=int, default=8082, help="WebSocket port")
    parser.add_argument("--dream-schedule", default="0 3 * * *", help="Dream schedule (cron)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent requests")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout (seconds)")
    parser.add_argument("--auto-restart", action="store_true", default=True, help="Auto-restart llama on failure")
    parser.add_argument("--restart-threshold", type=int, default=3, help="Failures before restart")
    
    args = parser.parse_args()
    
    # Create config
    config = create_daemon_config(
        workspace=args.workspace,
        llama_model=args.llama_model,
        llama_server=args.llama_server,
    )
    
    # Override with CLI args
    config.llama_port = args.llama_port
    config.llama_host = args.llama_host
    config.llama_ctx_size = args.llama_ctx_size
    config.llama_threads = args.llama_threads
    config.http_port = args.http_port
    config.http_host = args.http_host
    config.ws_port = args.ws_port
    config.dream_schedule = args.dream_schedule
    config.max_concurrent_requests = args.max_concurrent
    config.request_timeout = args.timeout
    config.auto_restart_llama = args.auto_restart
    config.llama_restart_threshold = args.restart_threshold
    
    print(f"Starting FRENESIS daemon...")
    print(f"  Workspace: {config.workspace}")
    print(f"  Model: {config.llama_model_path}")
    print(f"  llama-server: {config.llama_server_path}:{config.llama_port}")
    print(f"  Daemon HTTP: {config.http_host}:{config.http_port}")
    print(f"  Dream schedule: {config.dream_schedule}")
    print()
    
    # Create and run daemon
    daemon = FRENESISDaemon(config)
    
    try:
        asyncio.run(daemon.run_forever())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()