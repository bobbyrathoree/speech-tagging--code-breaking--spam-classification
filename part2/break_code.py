#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: PLEASE PUT YOUR NAMES AND USERIDS HERE
#
# based on skeleton code by D. Crandall, 11/2019
#
# ./break_code.py : attack encryption
#


import random
import math
import copy
import sys
import encode

# put your code here!
def break_code(string, corpus):
    return "Not implemented yet!"


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = encode.read_clean_file(sys.argv[1])
    corpus = encode.read_clean_file(sys.argv[2])
    decoded = break_code(encoded, corpus)

    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)
