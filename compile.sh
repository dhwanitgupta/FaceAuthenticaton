#! /bin/bash
g++ `pkg-config opencv --cflags` $1 `pkg-config opencv --libs` -g
