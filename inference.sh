#!/bin/bash
myloc="$(dirname "$0")"

python "$myloc/inference.py" -p "A cat holding a sign that says hello world" -hh 1024 -w 1024 -m 256 -n 25