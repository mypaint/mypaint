#!/bin/sh
(

#python -m profile -s cumulative mypaint
python -m profile -s time mypaint -p #dab_and_render_performance.myp

) > profile.out 2>&1
head -n 30 profile.out

