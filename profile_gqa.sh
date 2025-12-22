#!/bin/bash

set -e
set -o pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate py312

pip install nvtx

LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/home/tlwu/anaconda3/envs/py312/lib:/home/tlwu/cudnn9.8/lib64:/home/tlwu/cudnn9.8/lib

cd onnxruntime/test/python/transformers

rm -f gqa_fp16.nsys-rep
rm -f gqa_fp16.sqlite
nsys profile -o gqa_fp16 --export=sqlite python profile_gqa.py --mode fp16 --warmup 5 --repeat 100
python parse_nsys.py gqa_fp16.sqlite --skip-first 5 --tag Fp16

rm -f gqa_int8.nsys-rep
rm -f gqa_int8.sqlite
nsys profile -o gqa_int8 --export=sqlite python profile_gqa.py --mode int8 --warmup 5 --repeat 100
python parse_nsys.py gqa_int8.sqlite --skip-first 5 --tag Int8

rm -f gqa_int4.nsys-rep
rm -f gqa_int4.sqlite
nsys profile -o gqa_int4 --export=sqlite python profile_gqa.py --mode int4 --warmup 5 --repeat 100
python parse_nsys.py gqa_int4.sqlite --skip-first 5 --tag Int4

cd -
