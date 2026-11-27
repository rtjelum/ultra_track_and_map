import http.client
import json
import threading
import unittest
from unittest.mock import patch

import numpy as np

import audiodevice_server as server
import audiodevice_tracking as tracking


def fake_stream(serial, low, high, level, duration, stop_event, on_status, on_samples):
    on_status("Chirps active until Stop")
    fake_stream.started.set()
    stop_event.wait(5)
    raise InterruptedError("stopped")


class AudiodeviceServerTests(unittest.TestCase):
    def setUp(self):
        fake_stream.started = threading.Event()
        self.session = server.LiveSession("serial", (18000, 22000), 20, stream=fake_stream)
        self.httpd = server.make_server(self.session, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutting_down.set()
        self.session.stop()
        self.session.join(5)
        self.session.stop_lidar()
        self.session.join_lidar()
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, method, path, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        body = json.loads(response.read() or b"null")
        connection.close()
        return response.status, body

    def test_control_requires_custom_header_and_local_host(self):
        self.assertEqual(self.request("POST", "/api/start")[0], 403)
        status, _ = self.request("POST", "/api/start",
                                 {"X-Audiodevice-Control": "1", "Host": "evil.example:8765"})
        self.assertEqual(status, 403)
        self.assertIsNone(self.session.worker)

    def test_events_stream_status_and_frames_until_stop(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", "/events")
        events = connection.getresponse()

        def next_event():
            kind = data = None
            while True:
                line = events.fp.readline().decode().rstrip("\n")
                if line.startswith("event: "):
                    kind = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                elif not line and kind:
                    return kind, data

        self.assertEqual(next_event()[1]["running"], False)
        self.assertEqual(self.request("POST", "/api/start", {"X-Audiodevice-Control": "1"}),
                         (200, {"accepted": True}))
        self.assertTrue(fake_stream.started.wait(5))
        self.session._publish("frame", server.frame_payload(
            tracking.TrackingFrame("candidate", 9.0, 4, position=(1.0, -0.25), support=4)))
        seen = []
        while not seen or seen[-1][0] != "frame":
            seen.append(next_event())
        self.assertEqual(seen[-1][1]["position"], [1.0, -0.25])
        self.assertTrue(any(kind == "status" and data["running"] for kind, data in seen))

        self.request("POST", "/api/stop", {"X-Audiodevice-Control": "1"})
        while True:
            kind, data = next_event()
            if kind == "status" and not data["running"]:
                break
        self.assertEqual(data["text"], "Stopped; device cleanup finished.")
        connection.close()

    def test_frame_payload_normalizes_traces_and_keeps_score_grid(self):
        envelopes = np.zeros((tracking.PERIOD, 6))
        envelopes[10, :] = np.arange(1, 7)
        scores = np.ones((45, 61))
        payload = server.frame_payload(tracking.TrackingFrame(
            "no_target", 5.0, 4, envelopes, scores, 1.5, support=2))
        self.assertEqual(len(payload["traces"]), 6)
        self.assertEqual(max(payload["traces"][5]), 1.0)
        self.assertEqual(np.array(payload["scores"]).shape, (45, 61))
        json.dumps(payload)
        self.assertIsNone(payload["facing_wall_m"])
        walled = server.frame_payload(tracking.TrackingFrame("no_target", 5.0, 4, facing_wall_m=4.3214))
        self.assertEqual(walled["facing_wall_m"], 4.321)

    def test_raw_session_config_recalibration_and_frames(self):
        self.session.raw = True
        self.session.notch = (19_000,)
        config = self.session.config()
        self.assertTrue(config["raw"])
        self.assertEqual((len(config["depths"]), len(config["laterals"])), (45, 61))

        model = tracking.AudiodeviceTracker(raw=True, notch_hz=self.session.notch)
        raw = tracking.TrackingFrame("raw", 5.0, 0, np.ones((tracking.PERIOD, 6)),
                                     np.ones((45, 61)))
        with (patch.object(server.mapper, "live_model", return_value=model) as factory,
              patch.object(model, "push", return_value=[raw])):
            def stream(_serial, _low, _high, _level, _duration, _stop, _status, samples):
                samples(np.zeros((1, 6)))
            self.session.stream = stream
            self.session.worker = threading.current_thread()
            self.assertFalse(self.session.recalibrate())
            client = self.session.subscribe()
            self.session._run()
            events = [client.get_nowait() for _ in range(client.qsize())]
            self.session.unsubscribe(client)
        factory.assert_called_once_with(18000, 22000, notch_hz=(19000,), raw=True)
        payload = next(data for kind, data in events if kind == "frame")
        self.assertEqual(payload["state"], "raw")
        self.assertIsNone(payload["position"])
        self.assertIsNone(payload["threshold"])
        self.assertIn("Session finished", self.session.status["text"])

    def test_lidar_controls_and_replay_do_not_start_audio(self):
        self.session.lidar_port = "/dev/ttyUSB0"
        started = threading.Event()
        scan = {"points": [[90, 1.2, 180]], "hz": 10, "timestamp_ms": 123, "bad_packets": 0}

        def stream(serial, port, root, stop, on_scan):
            on_scan(scan)
            started.set()
            stop.wait(5)

        with patch.object(server.lidar, "stream", side_effect=stream):
            self.assertEqual(self.request("POST", "/api/lidar/start")[0], 403)
            self.assertEqual(self.request("POST", "/api/lidar/start",
                                         {"X-Audiodevice-Control": "1"}),
                             (200, {"accepted": True}))
            self.assertTrue(started.wait(5))
            self.assertIsNone(self.session.worker)
            client = self.session.subscribe()
            events = dict(client.get_nowait() for _ in range(client.qsize()))
            self.assertEqual(events["lidar"], scan)
            self.session.unsubscribe(client)
            self.request("POST", "/api/lidar/stop", {"X-Audiodevice-Control": "1"})
            self.session.join_lidar()
        self.assertFalse(self.session.lidar_status["running"])
        self.assertIsNone(self.session.last_scan)

    def test_lidar_failure_is_visible_and_allows_retry(self):
        self.session.lidar_port = "/dev/ttyUSB0"
        with patch.object(server.lidar, "stream", side_effect=OSError("permission denied")):
            self.assertTrue(self.session.start_lidar())
            self.session.join_lidar()
        self.assertTrue(self.session.lidar_status["error"])
        self.assertIn("permission denied", self.session.lidar_status["text"])
        self.assertIsNone(self.session.lidar_worker)


if __name__ == "__main__":
    unittest.main()
