#!/usr/bin/env python
"finish what h2def.py started"

filename = 'mydrawwidget.defs'

def run():
    caller_owns_return('get_nonwhite_as_pixbuf')
    caller_owns_return('get_as_pixbuf')
    null_ok('set_brush', 'brush')
    caller_owns_return('set_brush')


def reset():
    global i
    i = 0
def find(sub):
    global i, s
    i = s.find(sub, i)
def insert(sub):
    global i, s
    s = s[:i] + sub + s[i:]

def caller_owns_return(method):
    reset()
    find('(define-method ' + method)
    find('\n')
    insert('\n  (caller-owns-return #t)')

def null_ok(method, argname):
    reset()
    find('(define-method ' + method)
    find('(parameters')
    find('"%s")' % argname)
    find(')')
    insert(' (null-ok)')

s = open(filename).read()
run()
open(filename, 'w').write(s)
print 'corrected', filename
