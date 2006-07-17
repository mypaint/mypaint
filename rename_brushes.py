#!/usr/bin/env python
"Rename brushes/b003.myb and similar stuff to avoid conflict with user brushes"

import os, random
os.chdir('brushes')

def rename(old, new):
    os.system('svn mv "%s" "%s"' % (old, new))
    try:
        os.rename(old, new)
    except:
        pass

order = open('order.conf').read()
existing = [name[:-4] for name in os.listdir('.') if name.endswith('.myb')]
for name in existing:
    if not name.startswith('b') or name[1] not in '0123456789': continue
    oldname = name
    while name in existing:
        name = 'o%03d' % (random.randrange(700) + 300)
    print oldname, '==>', name
    rename(oldname + '.myb', name + '.myb')
    rename(oldname + '_prev.png', name + '_prev.png')
    order = order.replace(oldname, name)

open('order.conf', 'w').write(order)

