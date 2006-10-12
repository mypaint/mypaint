#!/usr/bin/env python
# gobjects need lots of scary repetitive code
# this script creates a new class from an old one
#
# usage:
# ./class-rename.py gtk_my_surface_old gtk_my_surface_composition

import sys, os
exists = os.path.exists
def copy(a, b): open(b, 'w').write(open(a).read())

trash, oldname, newname = sys.argv
oldname = oldname.split('_')
newname = newname.split('_')

filenames = []
for ext in ['.c', '.h']:
    old = ''.join(oldname) + ext
    new = ''.join(newname) + ext
    if not exists(new):
        assert exists(old), old + ' does not exist'
        copy(old, new)
    filenames.append(new)

#for filename in 
for filename in filenames:
    s = open(filename).read()

    def replace(old, new):
        global s
        print old, '==>', new
        s = s.replace(old, new)

    def replace_all(old, new):
        replace('_'.join([s.upper() for s in old]), '_'.join([s.upper() for s in new]))
        replace('_'.join([s.lower() for s in old]), '_'.join([s.lower() for s in new]))
        replace(''.join([s.capitalize() for s in old]), ''.join([s.capitalize() for s in new]))

    old = oldname[:] #copy
    new = newname[:] #copy
    replace_all(old, new)
    old.insert(1, 'is')
    new.insert(1, 'is')
    replace_all(old, new)
    old[1] = 'type'
    new[1] = 'type'
    replace_all(old, new)

    print 'writing', filename
    open(filename, 'w').write(s)

