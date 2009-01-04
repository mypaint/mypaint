#!/bin/sh
(

python -m cProfile -s time mypaint $@ #-p #dab_and_render_performance.myp
#python -m cProfile -o tmp.prof mypaint #-p #dab_and_render_performance.myp

) > profile.out 2>&1
head -n 30 profile.out

