#!/bin/sh
(

#python -m profile -s cumulative mypaint
python -m profile -s time mypaint

) > profile.out 2>&1
head -n 30 profile.out

