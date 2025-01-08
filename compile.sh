#!/bin/bash

myloc="$(dirname "$0")"

python "$myloc/text_encoder_1/compile.py"
python "$myloc/text_encoder_2/compile.py" -m 256
python "$myloc/transformer/compile.py" -hh 1024 -w 1024 -m 256
python "$myloc/decoder/compile.py" -hh 1024 -w 1024