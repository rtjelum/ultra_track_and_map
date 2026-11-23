#!/usr/bin/env python3
"""Stream the experimental live Audiodevice chirp tracker to a web browser."""

import argparse
import ipaddress
import json
import queue
import signal
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

import audiodevice_capture as capture
import audiodevice_live_audio as audio
import audiodevice_lidar as lidar
import audiodevice_tracking as tracker


PAGE = Path(__file__).resolve().parent / "audiodevice_server.html"
TRACE_COUNT = min(tracker.PERIOD, int(6 * capture.RATE / tracker.SPEED_OF_SOUND))
TRACE_BIN = 2
KEEPALIVE_SECONDS = 10
CONTROL_HEADER = "X-Audiodevice-Control"


def frame_payload(frame):
    """JSON-ready summary of a TrackingFrame for the browser."""
    payload = {
        "state": frame.state,
        "sync_ratio": round(frame.sync_ratio, 2),
        "calibration": frame.calibration,
        "support": frame.support,
        "threshold": None if frame.threshold is None else round(frame.threshold, 3),
        "position": None if frame.position is None else [round(v, 3) for v in frame.position],
        "traces": None,
        "scores": None,
    }
    if frame.envelopes is not None:
        envelopes = frame.envelopes[:TRACE_COUNT // TRACE_BIN * TRACE_BIN]
        envelopes = envelopes.reshape(-1, TRACE_BIN, 6).max(axis=1)
        maximum = envelopes.max(axis=0)
        envelopes = envelopes / np.where(maximum > 0, maximum, 1)
        payload["traces"] = np.round(envelopes.T, 3).tolist()
    if frame.scores is not None:
        payload["scores"] = np.round(frame.scores, 3).tolist()
    return payload


class LiveSession:
    """One device session at a time, broadcast to any number of browser clients."""

    def __init__(self, serial, band, level, duration=None, stream=None, *,
                 lidar_port=None, lidar_root=False, lidar_stream=None):
        self.serial = serial
        self.band = tuple(band)
        self.level = level
        self.duration = duration
        self.stream = stream or audio.stream
        self.lock = threading.Lock()
        self.clients = set()
        self.worker = None
        self.stopping = False
        self.stop_event = threading.Event()
        self.recalibrate_event = threading.Event()
        self.status = {"running": False, "stopping": False, "error": False,
                       "text": "Ready — press Start."}
        self.last_frame = None
        self._config = None
        self.lidar_port = lidar_port
        self.lidar_root = lidar_root
        self.lidar_stream = lidar_stream    # None: the device (lidar.stream)
        self.lidar_worker = None
        self.lidar_stop = threading.Event()
        self.lidar_status = {"running": False, "stopping": False, "error": False,
                             "text": "LD-19 ready." if lidar_port else "LiDAR not configured."}
        self.last_scan = None

    def config(self):
        if self._config is None:
            self._config = self._make_config()
        return self._config

    def _make_config(self):
        model = tracker.AudiodeviceTracker(*self.band)
        return {
            "serial": self.serial, "band": self.band, "level": self.level,
            "duration": self.duration,
            "depths": [round(v, 3) for v in model.depths],
            "laterals": [round(v, 3) for v in model.laterals],
            "mics": tracker.MIC_POSITIONS.tolist(),
            "speaker": tracker.TX_POSITION.tolist(),
            "calibration_updates": tracker.CALIBRATION_UPDATES,
            "trace_spacing_m": TRACE_BIN * tracker.SPEED_OF_SOUND / capture.RATE,
            "lidar_port": self.lidar_port,
        }

    def subscribe(self):
        client = queue.Queue(maxsize=16)
        with self.lock:
            self.clients.add(client)
            client.put_nowait(("status", self.status))
            client.put_nowait(("lidar_status", self.lidar_status))
            if self.last_frame is not None:
                client.put_nowait(("frame", self.last_frame))
            if self.last_scan is not None:
                client.put_nowait(("lidar", self.last_scan))
        return client

    def unsubscribe(self, client):
        with self.lock:
            self.clients.discard(client)

    def _publish(self, kind, data):
        with self.lock:
            if kind == "status":
                self.status = data
            elif kind == "frame":
                self.last_frame = data
            elif kind == "lidar_status":
                self.lidar_status = data
            elif kind == "lidar":
                self.last_scan = data
            for client in self.clients:
                # Slow clients skip stale updates instead of delaying the stream.
                while True:
                    try:
                        client.put_nowait((kind, data))
                        break
                    except queue.Full:
                        try:
                            client.get_nowait()
                        except queue.Empty:
                            pass

    def _set_status(self, text, *, error=False):
        with self.lock:
            running, stopping = self.worker is not None, self.stopping
        self._publish("status", {"running": running, "stopping": stopping,
                                 "error": error, "text": text})

    def start(self):
        with self.lock:
            if self.worker is not None:
                return False
            self.stop_event.clear()
            self.recalibrate_event.clear()
            self.stopping = False
            self.last_frame = None
            self.worker = threading.Thread(target=self._run, name="audiodevice-live", daemon=False)
        self._publish("reset", {})
        self._set_status("Preparing device — playback has not started yet.")
        self.worker.start()
        return True

    def stop(self):
        with self.lock:
            if self.worker is None:
                return False
            self.stopping = True
        self.stop_event.set()
        self._set_status("Stopping playback and restoring the device...")
        return True

    def recalibrate(self):
        with self.lock:
            if self.worker is None or self.stopping:
                return False
        self.recalibrate_event.set()
        self._publish("reset", {})
        self._set_status("Recalibrating — remove the target and keep the room still.")
        return True

    def join(self, timeout=None):
        worker = self.worker
        if worker is not None:
            worker.join(timeout)
        return worker is None or not worker.is_alive()

    def start_lidar(self):
        with self.lock:
            if not self.lidar_port or self.lidar_worker is not None:
                return False
            self.lidar_stop.clear()
            self.last_scan = None
            self.lidar_worker = threading.Thread(target=self._run_lidar, name="ld19", daemon=False)
        self._publish("lidar_status", {"running": True, "stopping": False, "error": False,
                                       "text": "Waiting for LD-19 scans..."})
        self.lidar_worker.start()
        return True

    def stop_lidar(self):
        with self.lock:
            if self.lidar_worker is None:
                return False
        self.lidar_stop.set()
        self._publish("lidar_status", {"running": True, "stopping": True, "error": False,
                                       "text": "Stopping LiDAR and restoring serial settings..."})
        return True

    def join_lidar(self):
        worker = self.lidar_worker
        if worker is not None:
            worker.join()

    def _run_lidar(self):
        text, error = "LiDAR stopped; serial settings restored.", False
        try:
            (self.lidar_stream or lidar.stream)(
                self.serial, self.lidar_port, self.lidar_root, self.lidar_stop,
                lambda scan: self._publish("lidar", scan))
        except (capture.CaptureError, subprocess.SubprocessError, OSError, ValueError) as exc:
            print(f"LiDAR failed: {exc}", file=sys.stderr)
            text, error = f"LiDAR error: {exc}", True
        except Exception as exc:
            print(f"Unexpected LiDAR failure: {exc!r}", file=sys.stderr)
            text, error = "Unexpected LiDAR failure; see the server terminal.", True
            raise
        finally:
            with self.lock:
                self.lidar_worker = None
                self.last_scan = None
            self._publish("lidar_status", {"running": False, "stopping": False,
                                           "error": error, "text": text})

    def _run(self):
        final, error = "", False
        try:
            model = tracker.AudiodeviceTracker(*self.band)
            synchronized = calibrated = False

            def on_status(text):
                if not self.stop_event.is_set():
                    self._set_status(text)

            def on_samples(samples):
                nonlocal synchronized, calibrated
                if self.recalibrate_event.is_set():
                    model.reset_calibration()
                    self.recalibrate_event.clear()
                    synchronized = calibrated = False
                for frame in model.push(samples):
                    if frame.state == "calibrating":
                        synchronized = True
                        calibrated |= frame.calibration == tracker.CALIBRATION_UPDATES
                    if not self.stop_event.is_set():
                        self._publish("frame", frame_payload(frame))

            self.stream(self.serial, *self.band, self.level, self.duration,
                        self.stop_event, on_status, on_samples)
            if not synchronized:
                final = "Session ended without a stable chirp reference — no position."
            elif not calibrated:
                final = "Session ended before background calibration completed."
            else:
                final = "Session finished; press Start to run again."
        except InterruptedError:
            final = "Stopped; device cleanup finished."
        except (capture.CaptureError, subprocess.SubprocessError, OSError, ValueError,
                RuntimeError) as exc:
            print(f"Live session failed: {exc}", file=sys.stderr)
            final, error = f"Error: {exc}", True
        except Exception as exc:
            print(f"Unexpected live session failure: {exc!r}", file=sys.stderr)
            final, error = "Unexpected session failure; see the server terminal.", True
            raise
        finally:
            with self.lock:
                self.worker = None
                self.stopping = False
                self.last_frame = None
            self._publish("reset", {})
            self._set_status(final, error=error)


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class Handler(BaseHTTPRequestHandler):
    server_version = "AudiodeviceLive/1"
    session = None
    allowed_hosts = frozenset()
    shutting_down = None

    def log_message(self, format, *args):
        pass

    def _host_allowed(self):
        # Reject DNS-rebinding requests whose Host header names another site.
        host = urlsplit("//" + (self.headers.get("Host") or "")).hostname or ""
        return is_loopback(host) or host in self.allowed_hosts

    def _send(self, status, body, content_type):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload), "application/json")

    def do_GET(self):
        if not self._host_allowed():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "Host not allowed"})
        path = urlsplit(self.path).path
        if path == "/":
            return self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/config":
            return self._json(HTTPStatus.OK, self.session.config())
        if path == "/events":
            return self._events()
        self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self):
        # A custom header cannot be sent cross-site without a CORS preflight,
        # which this server never approves, so other web pages cannot start audio.
        if not self._host_allowed() or self.headers.get(CONTROL_HEADER) != "1":
            return self._json(HTTPStatus.FORBIDDEN, {"error": "Control request not allowed"})
        actions = {"/api/start": self.session.start, "/api/stop": self.session.stop,
                   "/api/recalibrate": self.session.recalibrate,
                   "/api/lidar/start": self.session.start_lidar,
                   "/api/lidar/stop": self.session.stop_lidar}
        action = actions.get(urlsplit(self.path).path)
        if action is None:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        if self.shutting_down.is_set() and action in (self.session.start, self.session.start_lidar):
            return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Server is stopping"})
        self._json(HTTPStatus.OK, {"accepted": action()})

    def _events(self):
        client = self.session.subscribe()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while not self.shutting_down.is_set():
                try:
                    kind, data = client.get(timeout=KEEPALIVE_SECONDS)
                    message = f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                except queue.Empty:
                    message = ": keepalive\n\n"
                self.wfile.write(message.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.session.unsubscribe(client)


def make_server(session, host, port):
    shutting_down = threading.Event()
    handler = type("AudiodeviceHandler", (Handler,), {
        "session": session, "shutting_down": shutting_down,
        "allowed_hosts": frozenset({host}),
    })
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.shutting_down = shutting_down
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="ADB serial; auto-detect one connected Audiodevice if omitted")
    parser.add_argument("--band", type=float, nargs=2, default=(18_000, 22_000),
                        metavar=("LOW_HZ", "HIGH_HZ"))
    parser.add_argument("--level", type=float, default=20,
                        help="digital peak percentage (default 20; maximum 20)")
    parser.add_argument("--seconds", type=int,
                        help="optional session length in seconds (default: run until Stop)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (default 127.0.0.1, this machine only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--lidar-port", help="LD-19 serial port on target, e.g. /dev/ttyUSB0")
    parser.add_argument("--lidar-root", action="store_true",
                        help="use target su 0 for serial access; does not change permissions")
    parser.add_argument("--recording", metavar="STEM",
                        help="replay a recording (e.g. recordings/walk1) instead of a device")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="replay speed for --recording (default 1 = real time)")
    args = parser.parse_args()
    if not 0 < args.level <= 20:
        parser.error("--level must be greater than 0 and no more than 20")
    if args.seconds is not None and args.seconds < 5:
        parser.error("--seconds must be at least 5")
    try:
        if args.recording:
            import audiodevice_recording as recording
            player = recording.Player(args.recording, args.speed)
            serial = f"recording {args.recording}"
            session = LiveSession(serial, player.band, args.level, args.seconds, player.audio,
                                  lidar_port="recorded LD-19" if player.rec.scans else None,
                                  lidar_stream=player.lidar)
        else:
            capture.make_chirp(*args.band)
            serial = args.serial or capture.connected_serial()
            session = LiveSession(serial, args.band, args.level, args.seconds,
                                  lidar_port=args.lidar_port, lidar_root=args.lidar_root)
        server = make_server(session, args.host, args.port)
    except (capture.CaptureError, subprocess.SubprocessError, OSError, ValueError) as exc:
        print(f"Unable to start live server: {exc}", file=sys.stderr)
        return 1

    if not is_loopback(args.host) and not args.recording:
        print("WARNING: anyone who can reach this address can start device playback.",
              file=sys.stderr)
    host, port = server.server_address[:2]
    print(f"Audiodevice live server for {serial} at http://{'localhost' if is_loopback(host) else host}:{port}/",
          flush=True)
    print("Press Ctrl+C to stop the server and restore the device.", flush=True)

    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    thread = threading.Thread(target=server.serve_forever, name="http", daemon=True)
    thread.start()
    try:
        while not stop.wait(0.5):
            pass
    finally:
        server.shutting_down.set()
        if session.stop():
            print("Stopping playback and restoring the device...", file=sys.stderr)
        session.join()
        session.stop_lidar()
        session.join_lidar()
        server.shutdown()
        server.server_close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print("Server stopped; device cleanup finished.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
