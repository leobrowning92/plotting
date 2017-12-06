#!/usr/bin/env bash

## code for explicitly running ftp file transfer
# HOST='10.30.134.11'
# files
# ftp -n -v $HOST << EOT
# prompt
# user leo
# cd ram
# ls
# get test.txt
# bye
# EOT
$file="test.txt"
wget --no-passive ftp://name:leo@10.30.134.11/ram/test.txt
