"""FastAPI application factory."""
from __future__ import annotations

import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import analytics, charts, control, portfolio, sentiment, trades
from api.ws_hub import get_hub


def create_app() -> FastAPI:
    app = FastAPI(title="HellaCash Dashboard", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # REST routers
    app.include_router(trades.router)
    app.include_router(portfolio.router)
    app.include_router(analytics.router)
    app.include_router(sentiment.router)
    app.include_router(charts.router)
    app.include_router(control.router)

    # WebSocket live feed
    hub = get_hub()

    @app.on_event("startup")
    async def on_startup():
        await hub.start_relay()

    @app.websocket("/ws/feed")
    async def ws_feed(ws: WebSocket):
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keep-alive pings from client
        except WebSocketDisconnect:
            await hub.disconnect(ws)

    # Serve frontend static files
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.isdir(frontend_dir):
        app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")

        from fastapi.responses import FileResponse

        @app.get("/", include_in_schema=False)
        async def serve_dashboard():
            return FileResponse(os.path.join(frontend_dir, "index.html"))

    return app
