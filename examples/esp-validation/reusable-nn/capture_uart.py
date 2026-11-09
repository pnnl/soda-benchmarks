#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Record a ser2net Telnet endpoint without sending UART application data.

Run on the serial server with Python 3.12 or earlier (telnetlib). The output
must be fresh. The reader has a bounded lifetime and releases the connection
when interrupted; it never changes the serial server configuration.
"""
import argparse
import os
from pathlib import Path
import socket
import telnetlib
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("host")
parser.add_argument("port",type=int)
parser.add_argument("output",type=Path)
parser.add_argument("--duration",type=float,default=3600)
args=parser.parse_args()
assert args.duration>0
deadline=time.monotonic()+args.duration
with args.output.open("xb",buffering=0) as output:
    with telnetlib.Telnet(args.host,args.port,timeout=10) as connection:
        print(f"UART_CAPTURE_STARTED pid={os.getpid()} endpoint={args.host}:{args.port} log={args.output}",flush=True)
        while time.monotonic()<deadline:
            try:
                data=connection.read_some()
            except socket.timeout:
                continue
            if not data:
                raise EOFError("UART server closed the recording connection")
            output.write(data)
            if b"Port already in use" in data:
                raise RuntimeError("UART endpoint is already in use")
print("UART_CAPTURE_FINISHED",flush=True)
