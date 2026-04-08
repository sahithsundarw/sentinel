"""
Server entry point for multi-mode deployment (openenv serve / uv run).
Delegates to the main FastAPI application in app/main.py.
"""
import uvicorn


def main() -> None:
    """Start the Guardrail Arena server."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
