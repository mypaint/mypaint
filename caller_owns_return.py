#!/usr/bin/env python
"finish what h2def.py started"
import sys

filename = sys.argv[1]
modify_methods = sys.argv[2:]

old_lines = open(filename).readlines()
new_lines = []
while old_lines:
    line = old_lines.pop(0)
    new_lines.append(line)
    if line.startswith('(define-method '):
        method = line.split()[1]
        if method in modify_methods:
            new_lines.append('  (caller-owns-return #t)\n')
  
open(filename, 'w').write(''.join(new_lines))
print 'added caller-owns-return statement'
