"""Launch the loopback-only file UI; no microphone or automatic playback."""

from pathlib import Path
import sys
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    if sys.version_info < (3, 12):
        print("Python 3.12 or later is required.")
        return 1
    try:
        from hum_dsp.local_app import make_server
    except ImportError:
        print('Install dependencies first: python -m pip install .')
        return 1
    with make_server() as server:
        url = server.origin + "/"
        print("Hum DSP: files are processed on this computer only.", flush=True)
        print(f"Open this address in your browser: {url}", flush=True)
        print("Keep this window open. Press Ctrl+C here to stop.", flush=True)
        print("No automatic playback, microphone recording or external uploads.", flush=True)
        try:
            webbrowser.open(url)
        except webbrowser.Error:
            print("Open the address above manually.", flush=True)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
