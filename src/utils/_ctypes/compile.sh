#!/bin/bash
gcc -fPIC -I /usr/share/R/include -O3 coranking.cpp -shared -o coranking.so
