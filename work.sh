#!/bin/bash

for frac in 0 10 20 30; do
    python tools/brother3.py $frac
    time ./bin/main_debug $frac
done