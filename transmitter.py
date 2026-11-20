#!/usr/bin/env python3
"""
Ultrasonic transmitter — generates a 40 kHz square wave on GPIO18
using the Pi's hardware clock (zero jitter).

Requires pigpio daemon:  sudo pigpiod
"""

import pigpio
import time
import signal
import sys
import argparse

DEFAULT_FREQ = 20_000
DEFAULT_GPIO = 18       # hardware-clock-capable pins: 4,5,6,18,20,21,32,34,42,43,44,45


def main():
    parser = argparse.ArgumentParser(description="Ultrasonic transmitter via pigpio hardware clock")
    parser.add_argument("--freq", type=int, default=DEFAULT_FREQ, help="Transmit frequency in Hz (default 20000)")
    parser.add_argument("--gpio", type=int, default=DEFAULT_GPIO, help="GPIO pin (default 18)")
    args = parser.parse_args()

    pi = pigpio.pi()
    if not pi.connected:
        sys.exit("ERROR: pigpiod not running — start with:  sudo pigpiod")

    def shutdown(sig, frame):
        print("\nStopping transmitter...")
        pi.hardware_clock(args.gpio, 0)
        pi.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Transmitting {args.freq} Hz on GPIO{args.gpio}  —  Ctrl-C to stop")
    pi.hardware_clock(args.gpio, args.freq)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
