#!/usr/bin/env bash
# Run the Direct IP voice server (bind to all interfaces by default)
python3 direct_ip_call.py --mode server --ip 0.0.0.0 --port 50005
