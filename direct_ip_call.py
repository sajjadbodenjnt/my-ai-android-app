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


class DirectIPApp(App):
    def build(self):
        self.title = 'Direct IP Voice'
        return RootWidget()


if __name__ == '__main__':
    DirectIPApp().run()
