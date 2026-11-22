import os
import sys
import threading
import time
import webbrowser
from pathlib import Path


def resource_path(name):
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / name

    return Path(__file__).resolve().parent / name


def open_browser():
    time.sleep(3)
    webbrowser.open("http://localhost:8501")


def main():

    app_path = resource_path("app.py")

    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    threading.Thread(
        target=open_browser,
        daemon=True
    ).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=true",
        "--global.developmentMode=false"
    ]

    stcli.main()


if __name__ == "__main__":
    main()
