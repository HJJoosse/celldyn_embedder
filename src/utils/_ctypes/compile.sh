#!/bin/bash

gcc -fPIC -I /usr/share/R/include  -shared -o coranking.so coranking.cpp
