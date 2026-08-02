#!/usr/bin/env bash
# Run the Direct IP voice client. Replace TARGET_IP with peer IP.
if [ "$1" == "" ]; then
  echo "Usage: ./run_client.sh <TARGET_IP> [PORT]"
  exit 1
fi
PORT=${2:-50005}
python3 direct_ip_call.py --mode client --ip "$1" --port "$PORT"
