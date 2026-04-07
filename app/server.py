"""
Entry point for `server` console_scripts (used by openenv validate).
Starts the uvicorn server on port 7860.
"""
import uvicorn


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
