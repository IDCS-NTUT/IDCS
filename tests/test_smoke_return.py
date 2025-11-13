import socket
import threading
import time
import unittest

from tools import smoke_deepstream as smoke


class SmokeReturnFeedTests(unittest.TestCase):
    def test_is_local_address(self) -> None:
        self.assertTrue(smoke._is_local_address("127.0.0.1"))
        self.assertTrue(smoke._is_local_address("0.0.0.0"))
        self.assertTrue(smoke._is_local_address("localhost"))
        self.assertFalse(smoke._is_local_address("192.168.1.10"))

    def test_probe_captures_packets(self) -> None:
        host = "127.0.0.1"
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        sock.close()

        stop_event = threading.Event()

        def _sender() -> None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
                payload = b"1234567890"
                while not stop_event.is_set():
                    tx.sendto(payload, (host, port))
                    time.sleep(0.01)

        thread = threading.Thread(target=_sender, daemon=True)
        thread.start()
        try:
            stats = smoke._probe_return_feed(
                host,
                port,
                duration=0.5,
                bind_host=host,
                min_packets=2,
            )
        finally:
            stop_event.set()
            thread.join(timeout=2)

        self.assertGreaterEqual(stats["packets"], 2)
        self.assertGreater(stats["bytes"], 0)
        self.assertEqual(stats["port"], port)
        self.assertEqual(stats["host"], host)


if __name__ == "__main__":
    unittest.main()
