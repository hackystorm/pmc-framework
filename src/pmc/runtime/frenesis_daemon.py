# SPDX-License-Identifier: GPL-3.0-only
"""
FRENESIS Daemon - Always-on, event-driven agentic framework.

This module runs as a persistent daemon (Windows service / Linux systemd)
that manages:
- llama-server subprocess lifecycle
- Event-driven message routing (WebSocket, HTTP, internal queue)
- Scheduled dream cycles (cron-like)
- Tool call execution
- Memory operations
- External webhook handlers
- Dream cycle scheduling

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│  FRENESIS Daemon (Windows Service / systemd)                   │
├─────────────────────────────────────────────────────────────────┤
│  Event Loop (asyncio)                                           │
│  ├── llama-server Manager (subprocess lifecycle)               │
│  ├── Event Bus (asyncio.Queue + handlers)                      │
│  ├── HTTP/WebSocket Server (incoming events)                   │
│  ├── Scheduler (cron-style: dream cycles, memory ops)          │
│  ├── Tool Registry (memory.remember, restore, state, dream)    │
│  ├── Dream Scheduler (cron: 03:00 daily, or triggered)         │
│  └── FRENESIS Runtime (CompanionRuntime + ChatClient)          │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

# The framework's core is standard library only, and that is a non-negotiable
# constraint in `docs/lab.md`. The daemon's HTTP and WebSocket surface is the
# one part that wants a server, so `aiohttp` is an optional dependency declared
# as the `frenesis` extra rather than a hard one: importing this module must not
# require it, and only starting the HTTP surface does.
AIHTTP_MISSING = (
    "the FRENESIS HTTP/WebSocket surface needs the optional 'frenesis' "
    "dependency: pip install 'pmc-framework[frenesis]'. Everything else in "
    "this module runs without it."
)

try:  # pragma: no cover - presence of an optional dependency
    import aiohttp
    from aiohttp import web
except ImportError:  # pragma: no cover - absence of an optional dependency
    aiohttp = None
    web = None

# PMC imports. Relative, so this module imports from an installed package and
# from a checkout alike: a sys.path insert plus `src.pmc.*` only works before
# installation, and silently breaks after it.
from .client import ChatClient, ChatReply
from .providers import ExternalCall, ProviderRegistry, ResourceBudget
from .session import CompanionRuntime, Turn
from .tools import ToolRegistry

logger = logging.getLogger("frenesis_daemon")


def _require_http():
    """The aiohttp module, or a message naming how to get it.

    Called at the point the HTTP surface is built, not at import, so a machine
    without the optional extra can still use the daemon's other pieces.
    """
    if web is None:
        raise RuntimeError(AIHTTP_MISSING)
    return web


def configure_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure the daemon's logging, once, with an explicit destination.

    Logging is configured here rather than at import, for two reasons.
    Importing a module must not write a file into whatever the current working
    directory happens to be: that is how a stray log ends up in the repository
    root, and how this module's own history got one. And a caller that embeds
    the daemon in a larger process has to be able to keep its own logging
    configuration instead of having it replaced underneath it.
    """
    if logger.handlers:
        return
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        # Only written when a caller names a file.
        handlers.insert(0, logging.FileHandler(str(log_file)))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


@dataclass
class DaemonConfig:
    """Configuration for the FRENESIS daemon."""
    workspace: Path
    llama_model_path: Path
    llama_server_path: Path
    llama_port: int = 8080
    llama_host: str = "127.0.0.1"
    llama_ctx_size: int = 4096
    llama_threads: int = 8
    http_host: str = "127.0.0.1"
    http_port: int = 8081
    ws_port: int = 8082
    workspace_data: Path = field(init=False)
    dream_schedule: str = "0 3 * * *"  # Daily at 3 AM
    max_concurrent_requests: int = 4
    request_timeout: int = 300
    auto_restart_llama: bool = True
    llama_restart_threshold: int = 3  # failures before restart
    
    def __post_init__(self):
        self.workspace_data = self.workspace / "daemon_data"
        self.workspace_data.mkdir(parents=True, exist_ok=True)


@dataclass
class Event:
    """Event in the daemon's event bus."""
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    source: str = "daemon"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "source": self.source,
            "payload": self.payload
        }


class EventBus:
    """Async event bus for the daemon."""
    
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def subscribe(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Handler subscribed to {event_type}: {handler.__name__}")
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe a handler."""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)
    
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        await self._queue.put(event)
    
    async def start(self) -> None:
        """Start the event loop."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop the event loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Event bus stopped")
    
    async def _run(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                handlers = self._handlers.get(event.event_type, [])
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Handler error for {event.event_type}: {e}", exc_info=True)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event bus error: {e}", exc_info=True)


class LlamaServerManager:
    """Manages the llama-server subprocess lifecycle."""
    
    def __init__(self, config: DaemonConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.failure_count = 0
        self.start_time: Optional[float] = None
    
    async def start(self) -> bool:
        """Start llama-server subprocess."""
        if self.process and self.process.poll() is None:
            logger.info("llama-server already running")
            return True
        
        cmd = [
            str(self.config.llama_server_path),
            "-m", str(self.config.llama_model_path),
            "-c", str(self.config.llama_ctx_size),
            "--port", str(self.config.llama_port),
            "--host", self.config.llama_host,
            "--threads", str(self.config.llama_threads),
        ]
        
        logger.info(f"Starting llama-server: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            self.start_time = time.time()
            
            # Wait for server to be ready
            for attempt in range(30):
                await asyncio.sleep(1)
                if await self._health_check():
                    self.failure_count = 0
                    logger.info(f"llama-server ready on port {self.config.llama_port}")
                    return True
            
            logger.error("llama-server failed to start within timeout")
            await self.stop()
            return False
            
        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop llama-server subprocess."""
        if self.process and self.process.poll() is None:
            logger.info("Stopping llama-server...")
            try:
                if os.name == 'nt':
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.process.terminate()
                
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.process.wait), 
                        timeout=10
                    )
                except asyncio.TimeoutError:
                    self.process.kill()
                    await asyncio.to_thread(self.process.wait)
                
                logger.info("llama-server stopped")
            except Exception as e:
                logger.error(f"Error stopping llama-server: {e}")
        self.process = None
    
    async def _health_check(self) -> bool:
        """Check if llama-server is responding.

        An HTTP client is needed here, and that is the optional dependency. A
        missing one is reported as an unhealthy llama-server rather than raised:
        the daemon's automatic restart is the correct response either way, and
        this function's contract is a boolean, not an exception.
        """
        if aiohttp is None:
            logger.warning(AIHTTP_MISSING.strip())
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{self.config.llama_host}:{self.config.llama_port}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def health_check(self) -> bool:
        """Public health check with failure tracking."""
        healthy = await self._health_check()
        if not healthy:
            self.failure_count += 1
            logger.warning(f"llama-server health check failed ({self.failure_count}/{self.config.llama_restart_threshold})")
            if self.failure_count >= self.config.llama_restart_threshold and self.config.auto_restart_llama:
                logger.warning("Restarting llama-server due to repeated failures")
                await self.restart()
        else:
            self.failure_count = 0
        return healthy
    
    async def restart(self) -> bool:
        """Restart llama-server."""
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()
    
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class Scheduler:
    """Cron-style scheduler for recurring tasks."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.tasks: Dict[str, Dict] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def add_job(self, name: str, cron_expr: str, callback: Callable, **kwargs) -> None:
        """Add a cron-style job."""
        self.tasks[name] = {
            "cron": cron_expr,
            "callback": callback,
            "kwargs": kwargs,
            "last_run": None,
            "next_run": self._next_run(cron_expr)
        }
        logger.info(f"Scheduled job: {name} ({cron_expr})")
    
    def remove_job(self, name: str) -> bool:
        if name in self.tasks:
            del self.tasks[name]
            return True
        return False
    
    def _next_run(self, cron_expr: str) -> datetime:
        """Calculate next run time from cron expression.
        Supports standard 5-field cron: minute hour day month weekday
        """
        parts = cron_expr.split()
        if len(parts) != 5:
            return datetime.utcnow() + timedelta(hours=1)
        
        minute, hour, day, month, weekday = parts
        now = datetime.utcnow()
        
        # Simple approach: find next matching time within next 24 hours
        for i in range(24 * 60):  # Check up to 24 hours
            candidate = now + timedelta(minutes=i)
            if i == 0 and candidate <= now:
                continue
                
            # Check minute
            if minute != "*" and int(minute) != candidate.minute:
                continue
            # Check hour
            if hour != "*" and int(hour) != candidate.hour:
                continue
            # Check day of month
            if day != "*" and int(day) != candidate.day:
                continue
            # Check month
            if month != "*" and int(month) != candidate.month:
                continue
            # Check weekday (0=Sunday, 1=Monday, etc.)
            if weekday != "*" and int(weekday) != candidate.weekday() + 1:
                continue
            
            return candidate
        
        return datetime.utcnow() + timedelta(hours=1)
    
    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Scheduler started")
    
    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    async def _run(self) -> None:
        while self._running:
            now = datetime.utcnow()
            for name, task in self.tasks.items():
                if task["next_run"] <= now:
                    logger.info(f"Running scheduled job: {name}")
                    try:
                        callback = task["callback"]
                        if asyncio.iscoroutinefunction(callback):
                            await callback(**task["kwargs"])
                        else:
                            callback(**task["kwargs"])
                    except Exception as e:
                        logger.error(f"Job {name} failed: {e}", exc_info=True)
                    task["last_run"] = now
                    task["next_run"] = self._next_run(task["cron"])
            await asyncio.sleep(30)  # Check every 30 seconds


class FRENESISRuntime:
    """Wrapper around CompanionRuntime for daemon use."""
    
    def __init__(self, config: DaemonConfig):
        self.config = config
        self.runtime: Optional[CompanionRuntime] = None
        self.client: Optional[ChatClient] = None
        self.instance_id: Optional[str] = None
    
    async def initialize(self) -> bool:
        """Initialize the FRENESIS runtime."""
        try:
            self.client = ChatClient(
                endpoint=f"http://{self.config.llama_host}:{self.config.llama_port}",
                model="frenesis",
                timeout=self.config.request_timeout,
            )
            
            self.runtime = CompanionRuntime(
                workspace=self.config.workspace,
                client=self.client,
            )
            self.instance_id = self.runtime.instance_id
            logger.info(f"FRENESIS runtime initialized: {self.instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize FRENESIS runtime: {e}")
            return False
    
    async def ask(self, question: str, with_state: bool = True) -> ChatReply:
        """Ask FRENESIS a question."""
        if not self.runtime:
            raise RuntimeError("Runtime not initialized")
        return self.runtime.ask(question, with_state=with_state)
    
    async def talk(self, text: str, with_state: bool = True) -> Turn:
        """Talk to FRENESIS and remember the exchange."""
        if not self.runtime:
            raise RuntimeError("Runtime not initialized")
        return self.runtime.talk(text, with_state=with_state)
    
    async def remember(self, text: str, load: float = 0.3, topics: List[str] = None) -> str:
        """Store a memory."""
        if not self.runtime:
            raise RuntimeError("Runtime not initialized")
        return self.runtime.remember(text, load=load, topics=topics or [])
    
    async def dream(self) -> Dict[str, Any]:
        """Trigger a dream cycle."""
        # This would invoke the dream pipeline
        # For now, return placeholder
        return {"status": "dream_triggered", "timestamp": datetime.utcnow().isoformat()}
    
    async def describe(self) -> Dict[str, Any]:
        """Get instance description."""
        if not self.runtime:
            return {}
        return self.runtime.describe()


class HTTPServer:
    """HTTP/WebSocket server for external events."""
    
    def __init__(self, config: DaemonConfig, event_bus: EventBus, runtime: FRENESISRuntime):
        # The one place the optional dependency is required, checked here so
        # the failure names the fix instead of an AttributeError deep in aiohttp.
        _require_http()
        self.config = config
        self.event_bus = event_bus
        self.runtime = runtime
        self.app = web.Application()
        self.setup_routes()
        self.runner: Optional[web.AppRunner] = None
    
    def setup_routes(self) -> None:
        self.app.router.add_get("/health", self.health)
        self.app.router.add_post("/api/v1/ask", self.ask)
        self.app.router.add_post("/api/v1/talk", self.talk)
        self.app.router.add_post("/api/v1/remember", self.remember)
        self.app.router.add_post("/api/v1/dream", self.dream)
        self.app.router.add_get("/api/v1/state", self.get_state)
        self.app.router.add_post("/api/v1/event", self.receive_event)
        self.app.router.add_get("/ws", self.websocket_handler)
    
    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "service": "frenesis-daemon"})
    
    async def ask(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            question = data.get("question", "")
            with_state = data.get("with_state", True)
            reply = await self.runtime.ask(question, with_state=with_state)
            return web.json_response({
                "answer": reply.text,
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "finish_reason": reply.finish_reason
            })
        except Exception as e:
            logger.error(f"Ask error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def talk(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            text = data.get("text", "")
            with_state = data.get("with_state", True)
            turn = await self.runtime.talk(text, with_state=with_state)
            return web.json_response({
                "user": turn.user,
                "assistant": turn.assistant,
                "record_id": turn.record_id
            })
        except Exception as e:
            logger.error(f"Talk error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def remember(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            text = data.get("text", "")
            load = data.get("load", 0.3)
            topics = data.get("topics", [])
            record_id = await self.runtime.remember(text, load=load, topics=topics)
            return web.json_response({"record_id": record_id})
        except Exception as e:
            logger.error(f"Remember error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def dream(self, request: web.Request) -> web.Response:
        try:
            result = await self.runtime.dream()
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Dream error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_state(self, request: web.Request) -> web.Response:
        try:
            state = await self.runtime.describe()
            return web.json_response(state)
        except Exception as e:
            logger.error(f"State error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def receive_event(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            event = Event(
                event_type=data.get("event_type", "external"),
                payload=data.get("payload", {}),
                correlation_id=data.get("correlation_id"),
                source=data.get("source", "http")
            )
            await self.event_bus.publish(event)
            return web.json_response({"status": "accepted", "event_id": event.event_id})
        except Exception as e:
            logger.error(f"Event error: {e}")
            return web.json_response({"error": str(e)}, status=400)
    
    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        logger.info("WebSocket connection established")
        
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    event = Event(
                        event_type=data.get("event_type", "websocket"),
                        payload=data.get("payload", {}),
                        correlation_id=data.get("correlation_id"),
                        source="websocket"
                    )
                    await self.event_bus.publish(event)
                    
                    # Echo back acknowledgment
                    await ws.send_json({
                        "type": "ack",
                        "event_id": event.event_id
                    })
                except Exception as e:
                    logger.error(f"WS error: {e}")
                    await ws.send_json({"error": str(e)})
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WS error: {ws.exception()}")
        
        logger.info("WebSocket connection closed")
        return ws
    
    async def start(self) -> None:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.config.http_host, self.config.http_port)
        await site.start()
        logger.info(f"HTTP server started on http://{self.config.http_host}:{self.config.http_port}")
    
    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()
            logger.info("HTTP server stopped")


class FRENESISDaemon:
    """Main daemon orchestrating all components."""
    
    def __init__(self, config: DaemonConfig):
        self.config = config
        self.event_bus = EventBus()
        self.llama_manager = LlamaServerManager(config)
        self.scheduler = Scheduler(self.event_bus)
        self.runtime = FRENESISRuntime(config)
        self.http_server = HTTPServer(config, self.event_bus, self.runtime)
        self._running = False
        self._main_task: Optional[asyncio.Task] = None
        
        # Setup default event handlers
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup default event handlers."""
        
        async def on_external_event(event: Event):
            logger.info(f"External event: {event.event_type} - {event.payload}")
            # Could trigger automatic responses, memories, etc.
        
        async def on_dream_triggered(event: Event):
            logger.info("Dream cycle triggered")
            result = await self.runtime.dream()
            logger.info(f"Dream completed: {result}")
        
        async def on_memory_requested(event: Event):
            """Handle memory.remember requests from external sources."""
            payload = event.payload
            text = payload.get("text", "")
            load = payload.get("load", 0.3)
            topics = payload.get("topics", [])
            if text:
                record_id = await self.runtime.remember(text, load=load, topics=topics)
                logger.info(f"External memory stored: {record_id}")
        
        async def on_dream_scheduled(event: Event):
            """Triggered by scheduler for dream cycles."""
            logger.info("Scheduled dream cycle starting...")
            result = await self.runtime.dream()
            logger.info(f"Scheduled dream completed: {result}")
        
        self.event_bus.subscribe("external", on_external_event)
        self.event_bus.subscribe("dream.trigger", on_dream_triggered)
        self.event_bus.subscribe("memory.remember", on_memory_requested)
        self.event_bus.subscribe("dream.scheduled", on_dream_scheduled)
    
    async def start(self) -> bool:
        """Start the daemon."""
        logger.info("Starting FRENESIS daemon...")
        
        # Start llama-server
        if not await self.llama_manager.start():
            logger.error("Failed to start llama-server")
            return False
        
        # Initialize FRENESIS runtime
        if not await self.runtime.initialize():
            logger.error("Failed to initialize FRENESIS runtime")
            await self.llama_manager.stop()
            return False
        
        # Start event bus
        await self.event_bus.start()
        
        # Start scheduler with dream cycle
        self.scheduler.add_job(
            "dream_cycle",
            self.config.dream_schedule,
            self._trigger_dream
        )
        await self.scheduler.start()
        
        # Start HTTP server
        await self.http_server.start()
        
        # Start health monitoring
        self._running = True
        self._main_task = asyncio.create_task(self._health_monitor())
        
        logger.info("FRENESIS daemon started successfully")
        return True
    
    async def _trigger_dream(self) -> None:
        """Trigger a dream cycle."""
        await self.event_bus.publish(Event(
            event_type="dream.scheduled",
            payload={},
            source="scheduler"
        ))
    
    async def _health_monitor(self) -> None:
        """Monitor health of all components."""
        while self._running:
            try:
                # Check llama-server health
                await self.llama_manager.health_check()
                
                # Check runtime
                if self.runtime.runtime:
                    state = await self.runtime.describe()
                    logger.debug(f"FRENESIS state: {state}")
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        logger.info("Stopping FRENESIS daemon...")
        self._running = False
        
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        
        await self.http_server.stop()
        await self.scheduler.stop()
        await self.event_bus.stop()
        await self.llama_manager.stop()
        
        logger.info("FRENESIS daemon stopped")
    
    async def run_forever(self) -> None:
        """Run until stopped."""
        if not await self.start():
            logger.error("Failed to start daemon")
            return
        
        logger.info("FRENESIS daemon running. Press Ctrl+C to stop.")
        
        try:
            while self._running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()


# Windows Service wrapper
class WindowsService:
    """Windows Service wrapper for the daemon."""
    
    def __init__(self, config: DaemonConfig):
        self.config = config
        self.daemon: Optional[FRENESISDaemon] = None
    
    def run(self) -> None:
        """Run as Windows service."""
        import asyncio
        
        async def run_daemon():
            self.daemon = FRENESISDaemon(self.config)
            await self.daemon.run_forever()
        
        asyncio.run(run_daemon())


def create_daemon_config(
    workspace: str,
    llama_model: str,
    llama_server: str,
    **kwargs
) -> DaemonConfig:
    """Factory function to create daemon config."""
    return DaemonConfig(
        workspace=Path(workspace),
        llama_model_path=Path(llama_model),
        llama_server_path=Path(llama_server),
        **kwargs
    )


async def main():
    """Main entry point for running as standalone daemon."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FRENESIS Daemon")
    parser.add_argument("--workspace", required=True, help="FRENESIS workspace path")
    parser.add_argument("--llama-model", required=True, help="Path to GGUF model")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server executable")
    parser.add_argument("--llama-port", type=int, default=8080, help="llama-server port")
    parser.add_argument("--http-port", type=int, default=8081, help="Daemon HTTP port")
    parser.add_argument("--dream-schedule", default="0 3 * * *", help="Dream schedule (cron)")
    parser.add_argument("--service", action="store_true", help="Run as Windows service")
    
    args = parser.parse_args()

    # Logging is set up here, when something is actually run, rather than at
    # import time: see configure_logging.
    configure_logging(
        log_file=Path(args.workspace) / "daemon_data" / "frenesis_daemon.log",
        level=logging.INFO,
    )

    config = DaemonConfig(
        workspace=Path(args.workspace),
        llama_model_path=Path(args.llama_model),
        llama_server_path=Path(args.llama_server),
        llama_port=args.llama_port,
        http_port=args.http_port,
        dream_schedule=args.dream_schedule,
    )
    
    if args.service and os.name == 'nt':
        service = WindowsService(config)
        service.run()
    else:
        daemon = FRENESISDaemon(config)
        await daemon.run_forever()


if __name__ == "__main__":
    asyncio.run(main())