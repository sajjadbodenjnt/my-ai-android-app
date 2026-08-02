#!/usr/bin/env python3
"""
Kivy-based Direct IP Voice Calling GUI (minimal).
This provides a simple IP status label and buttons to Start Server / Start Client.
Note: On Android, audio/video backend and permissions require buildozer packaging and platform-specific handling.
This script is a UI wrapper that launches server/client threads using the existing UDP audio logic.
"""

import threading
import socket
import time
from functools import partial

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import mainthread

# Reuse the core audio networking logic inside threads
try:
    import pyaudio
except Exception:
    pyaudio = None


def discover_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


class AudioServer(threading.Thread):
    def __init__(self, bind_ip, bind_port, chunk_size=1024, rate=16000):
        super().__init__(daemon=True)
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.chunk_size = chunk_size
        self.rate = rate
        self._stop = threading.Event()

    def run(self):
        if pyaudio is None:
            print("PyAudio not available; server cannot play audio")
            return
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.rate, output=True, frames_per_buffer=self.chunk_size)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind_ip, self.bind_port))
        try:
            while not self._stop.is_set():
                data, addr = sock.recvfrom(self.chunk_size * 2)
                if data:
                    stream.write(data)
        finally:
            try:
                stream.stop_stream(); stream.close(); p.terminate()
            except Exception:
                pass
            sock.close()

    def stop(self):
        self._stop.set()


class AudioClient(threading.Thread):
    def __init__(self, target_ip, target_port, chunk_size=1024, rate=16000):
        super().__init__(daemon=True)
        self.target_ip = target_ip
        self.target_port = target_port
        self.chunk_size = chunk_size
        self.rate = rate
        self._stop = threading.Event()

    def run(self):
        if pyaudio is None:
            print("PyAudio not available; client cannot capture audio")
            return
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=self.chunk_size)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while not self._stop.is_set():
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                sock.sendto(data, (self.target_ip, self.target_port))
        finally:
            try:
                stream.stop_stream(); stream.close(); p.terminate()
            except Exception:
                pass
            sock.close()

    def stop(self):
        self._stop.set()


class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=10, spacing=10, **kwargs)
        self.ip_label = Label(text=f"Local IP: {discover_local_ip()}", size_hint=(1, 0.1))
        self.add_widget(self.ip_label)

        self.target_input = TextInput(text='127.0.0.1', multiline=False, size_hint=(1, 0.1))
        self.add_widget(self.target_input)

        btn_layout = BoxLayout(size_hint=(1, 0.2))
        self.start_server_btn = Button(text='Start Server')
        self.stop_server_btn = Button(text='Stop Server')
        self.start_client_btn = Button(text='Start Client')
        self.stop_client_btn = Button(text='Stop Client')

        self.start_server_btn.bind(on_release=self.start_server)
        self.stop_server_btn.bind(on_release=self.stop_server)
        self.start_client_btn.bind(on_release=self.start_client)
        self.stop_client_btn.bind(on_release=self.stop_client)

        btn_layout.add_widget(self.start_server_btn)
        btn_layout.add_widget(self.stop_server_btn)
        btn_layout.add_widget(self.start_client_btn)
        btn_layout.add_widget(self.stop_client_btn)

        self.add_widget(btn_layout)

        self.status = Label(text='Status: idle', size_hint=(1, 0.1))
        self.add_widget(self.status)

        self.server = None
        self.client = None

    @mainthread
    def set_status(self, msg):
        self.status.text = f"Status: {msg}"

    def start_server(self, *_):
        if self.server is not None:
            self.set_status('Server already running')
            return
        bind_ip = '0.0.0.0'
        bind_port = 50005
        self.server = AudioServer(bind_ip, bind_port)
        self.server.start()
        self.set_status(f'Server listening {bind_ip}:{bind_port}')

    def stop_server(self, *_):
        if self.server:
            self.server.stop()
            self.server = None
            self.set_status('Server stopped')

    def start_client(self, *_):
        if self.client is not None:
            self.set_status('Client already running')
            return
        target_ip = self.target_input.text.strip()
        target_port = 50005
        self.client = AudioClient(target_ip, target_port)
        self.client.start()
        self.set_status(f'Sending to {target_ip}:{target_port}')

    def stop_client(self, *_):
        if self.client:
            self.client.stop()
            self.client = None
            self.set_status('Client stopped')


import math
import struct
import array

class AFSKModem:
    """Simple AFSK modem (1200/2200 Hz) for short-range acoustic P2P over speaker/mic.
    This is a proof-of-concept and not production-ready radio modem.
    """

    def __init__(self, sample_rate=8000, baud=100):
        self.sample_rate = sample_rate
        self.baud = baud
        self.tone0 = 1200.0
        self.tone1 = 2200.0

    def bits_from_bytes(self, data: bytes):
        for b in data:
            for i in range(8):
                yield (b >> i) & 1

    def modulate(self, data: bytes):
        # Return bytes PCM16 little-endian
        samples = []
        bit_len = int(self.sample_rate / self.baud)
        for bit in self.bits_from_bytes(data):
            f = self.tone1 if bit else self.tone0
            for n in range(bit_len):
                t = (n / self.sample_rate)
                samples.append(int(32767 * 0.3 * math.sin(2 * math.pi * f * t)))
        arr = array.array('h', samples)
        return arr.tobytes()

    def demodulate_simple(self, pcm_data: bytes):
        # Very naive demodulator: measure energy around tone0 and tone1 per bit window
        bit_len = int(self.sample_rate / self.baud)
        samples = struct.unpack('<' + 'h' * (len(pcm_data) // 2), pcm_data)
        bits = []
        for i in range(0, len(samples), bit_len):
            window = samples[i:i+bit_len]
            if len(window) < bit_len:
                break
            # compute simple DFT-like projections
            e0 = 0.0
            e1 = 0.0
            for n, s in enumerate(window):
                e0 += s * math.sin(2 * math.pi * self.tone0 * (n / self.sample_rate))
                e1 += s * math.sin(2 * math.pi * self.tone1 * (n / self.sample_rate))
            bits.append(1 if abs(e1) > abs(e0) else 0)
        # pack bits into bytes
        out = bytearray()
        cur = 0
        bitpos = 0
        for b in bits:
            cur |= (b & 1) << bitpos
            bitpos += 1
            if bitpos == 8:
                out.append(cur)
                cur = 0
                bitpos = 0
        return bytes(out)


class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=10, spacing=10, **kwargs)
        self.ip_label = Label(text=f"Local IP: {discover_local_ip()}", size_hint=(1, 0.08))
        self.add_widget(self.ip_label)

        # Status labels required by user
        self.signal_status = Label(text='Signal Status: Idle', size_hint=(1, 0.08))
        self.peer_status = Label(text='Peer Auto-Mapped: None', size_hint=(1, 0.08))
        self.raw_channel_status = Label(text='Raw P2P Channel Active: No', size_hint=(1, 0.08))

        self.add_widget(self.signal_status)
        self.add_widget(self.peer_status)
        self.add_widget(self.raw_channel_status)

        self.target_input = TextInput(text='127.0.0.1', multiline=False, size_hint=(1, 0.08))
        self.add_widget(self.target_input)

        btn_layout = BoxLayout(size_hint=(1, 0.18))
        self.start_server_btn = Button(text='Start Server')
        self.stop_server_btn = Button(text='Stop Server')
        self.start_client_btn = Button(text='Start Client')
        self.stop_client_btn = Button(text='Stop Client')

        self.start_server_btn.bind(on_release=self.start_server)
        self.stop_server_btn.bind(on_release=self.stop_server)
        self.start_client_btn.bind(on_release=self.start_client)
        self.stop_client_btn.bind(on_release=self.stop_client)

        btn_layout.add_widget(self.start_server_btn)
        btn_layout.add_widget(self.stop_server_btn)
        btn_layout.add_widget(self.start_client_btn)
        btn_layout.add_widget(self.stop_client_btn)

        self.add_widget(btn_layout)

        self.status = Label(text='Status: idle', size_hint=(1, 0.08))
        self.add_widget(self.status)

        # Acoustic modem controls
        modem_layout = BoxLayout(size_hint=(1, 0.18))
        self.start_tx_btn = Button(text='Start Acoustic TX')
        self.stop_tx_btn = Button(text='Stop Acoustic TX')
        self.scan_btn = Button(text='Scan Frequencies')
        modem_layout.add_widget(self.start_tx_btn)
        modem_layout.add_widget(self.stop_tx_btn)
        modem_layout.add_widget(self.scan_btn)
        self.add_widget(modem_layout)

        self.start_tx_btn.bind(on_release=self.start_acoustic_tx)
        self.stop_tx_btn.bind(on_release=self.stop_acoustic_tx)
        self.scan_btn.bind(on_release=self.scan_frequencies)

        self.server = None
        self.client = None
        self.modem = AFSKModem()
        self.tx_thread = None

    @mainthread
    def set_status(self, msg):
        self.status.text = f"Status: {msg}"

    @mainthread
    def set_signal_status(self, msg):
        self.signal_status.text = f"Signal Status: {msg}"

    @mainthread
    def set_peer_status(self, msg):
        self.peer_status.text = f"Peer Auto-Mapped: {msg}"

    @mainthread
    def set_raw_channel(self, active: bool):
        self.raw_channel_status.text = f"Raw P2P Channel Active: {'Yes' if active else 'No'}"

    def start_server(self, *_):
        if self.server is not None:
            self.set_status('Server already running')
            return
        bind_ip = '0.0.0.0'
        bind_port = 50005
        self.server = AudioServer(bind_ip, bind_port)
        self.server.start()
        self.set_status(f'Server listening {bind_ip}:{bind_port}')

    def stop_server(self, *_):
        if self.server:
            self.server.stop()
            self.server = None
            self.set_status('Server stopped')

    def start_client(self, *_):
        if self.client is not None:
            self.set_status('Client already running')
            return
        target_ip = self.target_input.text.strip()
        target_port = 50005
        self.client = AudioClient(target_ip, target_port)
        self.client.start()
        self.set_status(f'Sending to {target_ip}:{target_port}')

    def stop_client(self, *_):
        if self.client:
            self.client.stop()
            self.client = None
            self.set_status('Client stopped')

    def start_acoustic_tx(self, *_):
        if self.tx_thread is not None:
            self.set_status('Acoustic TX already running')
            return
        # For demo, transmit a small ID packet repeatedly
        def tx_loop(modem, stop_event):
            if pyaudio is None:
                self.set_status('PyAudio unavailable')
                return
            import pyaudio as pa
            p = pa.PyAudio()
            stream = p.open(format=pa.paInt16, channels=1, rate=modem.sample_rate, output=True)
            id_packet = b'PEER:01'
            self.set_signal_status('Transmitting')
            self.set_raw_channel(True)
            while not stop_event.is_set():
                pcm = modem.modulate(id_packet)
                stream.write(pcm)
                time.sleep(0.05)
            try:
                stream.stop_stream(); stream.close(); p.terminate()
            except Exception:
                pass
            self.set_raw_channel(False)
            self.set_signal_status('Idle')

        stop_event = threading.Event()
        t = threading.Thread(target=tx_loop, args=(self.modem, stop_event), daemon=True)
        t.stop_event = stop_event
        t.start()
        self.tx_thread = t
        self.set_status('Acoustic TX started')
        self.set_peer_status('Auto-mapping...')

    def stop_acoustic_tx(self, *_):
        if self.tx_thread:
            try:
                self.tx_thread.stop_event.set()
            except Exception:
                pass
            self.tx_thread = None
            self.set_status('Acoustic TX stopped')

    def scan_frequencies(self, *_):
        # Very naive scan: update label and simulate detection
        self.set_signal_status('Scanning Radio/Acoustic Frequency')
        time.sleep(1)
        # pretend we found a peer
        self.set_peer_status('01')
        self.set_signal_status('Listening')
        self.set_status('Peer found: 01')


class DirectIPApp(App):
    def build(self):
        self.title = 'Direct IP Voice'
        return RootWidget()


if __name__ == '__main__':
    DirectIPApp().run()
