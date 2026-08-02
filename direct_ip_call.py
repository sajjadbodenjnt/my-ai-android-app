#!/usr/bin/env python3
"""
Simple Direct IP-to-IP voice call demo using UDP and PyAudio.
Modes:
 - server: listens on UDP port and plays incoming audio
 - client: captures microphone and sends to target IP:port
This is a minimal proof-of-concept for local network use only.
"""

import argparse
import socket
import threading
import time


def start_server(bind_ip: str, bind_port: int, chunk_size: int = 1024, rate: int = 16000):
    import pyaudio

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True, frames_per_buffer=chunk_size)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_ip, bind_port))
    print(f"Server listening on {bind_ip}:{bind_port}")

    try:
        while True:
            data, addr = sock.recvfrom(chunk_size * 2)
            if not data:
                continue
            stream.write(data)
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()


def start_client(target_ip: str, target_port: int, chunk_size: int = 1024, rate: int = 16000):
    import pyaudio

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk_size)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Sending audio to {target_ip}:{target_port} (press Ctrl-C to stop)")

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            sock.sendto(data, (target_ip, target_port))
    except KeyboardInterrupt:
        print("Stopping client...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()


def discover_local_ip():
    # Quick method to get the local IP address without external requests
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["server", "client"], required=True)
    p.add_argument("--ip", default="0.0.0.0", help="bind IP for server / target IP for client")
    p.add_argument("--port", type=int, default=50005)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--rate", type=int, default=16000)
    args = p.parse_args()

    if args.mode == "server":
        print("Local IP (for clients):", discover_local_ip())
        start_server(args.ip, args.port, args.chunk_size, args.rate)
    else:
        start_client(args.ip, args.port, args.chunk_size, args.rate)


if __name__ == "__main__":
    main()
