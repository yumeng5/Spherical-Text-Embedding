#!/usr/bin/env bash

cd src
gcc jose.c -o jose -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
