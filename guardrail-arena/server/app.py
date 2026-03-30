"""
Server entry point for multi-mode deployment (openenv serve / uv run server).
Launches the Guardrail Arena FastAPI app via uvicorn.
"""
import os
import uvicorn


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("app.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
