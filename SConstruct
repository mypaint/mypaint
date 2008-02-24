import os
from subprocess import Popen, PIPE

profile = "" # -g -pg

env = Environment(ENV = os.environ)
env.ParseConfig('pkg-config --cflags --libs gtk+-2.0 pygtk-2.0')
env.ParseConfig('python-config --cflags --ldflags')

env.Command('brushsettings.h', ['generate.py', 'brushsettings.py'], './generate.py')

defsdir = Popen(['pkg-config', 'pygtk-2.0', '--variable=defsdir'], stdout=PIPE).communicate()[0].strip()
assert defsdir, 'you need pygtk'

pygtk_codegen = 'pygtk-codegen-2.0'
pygtk_h2def = '/usr/share/pygtk/2.0/codegen/h2def.py'
glib_genmarshal = 'glib-genmarshal'

env.Command('mydrawwidget.defs.c', ['mydrawwidget.defs', 'mydrawwidget.override'], '''
%s --prefix mydrawwidget \
--register %s/gdk-types.defs \
--register %s/gtk-types.defs \
--override mydrawwidget.override \
mydrawwidget.defs > mydrawwidget.defs.c
''' % (pygtk_codegen, defsdir, defsdir))

defs_inputs = 'gtkmydrawwidget.h gtkmybrush.h gtkmysurface.h gtkmysurfaceold.h'
env.Command('mydrawwidget.defs', Split(defs_inputs + ' fix_generated_defs.py'), '''
python %s %s > mydrawwidget.defs && ./fix_generated_defs.py
''' % (pygtk_h2def, defs_inputs))

env.Command('mymarshal.h', 'mymarshal.list', glib_genmarshal + ' --prefix=mymarshal --header $SOURCE > $TARGET')
env.Command('mymarshal.c', 'mymarshal.list', glib_genmarshal + ' --prefix=mymarshal --body   $SOURCE > $TARGET')

sources = '''
mydrawwidget.defs.c
mydrawwidgetmodule.c 
gtkmydrawwidget.c 
gtkmybrush.c 
gtkmysurface.c 
gtkmysurfaceold.c 
brush_dab.c 
helpers.c 
mapping.c 
mymarshal.c 
lfd.c 
gestures.c 
stroke_recorder.c
'''

#env.SharedLibrary('mydrawwidget', Split(sources))
env.LoadableModule('mydrawwidget', Split(sources),
                   LDMODULEPREFIX='',
                   )

# gtkmydrawwidget.o:	brushsettings.h gtkmydrawwidget.c gtkmydrawwidget.h
# 	cc $(CFLAGS) -c -o $@ gtkmydrawwidget.c
# env.

# --- cut here ---

# PROFILE = #-g #-pg
# CFLAGS = $(PROFILE) -O3 `pkg-config --cflags gtk+-2.0 pygtk-2.0` -Wall -Werror -I/usr/include/python2.3/ -I.
# LDFLAGS = $(PROFILE) -O3 `pkg-config --libs gtk+-2.0 pygtk-2.0` -Wall -Werror
# DEFSDIR = `pkg-config --variable=defsdir pygtk-2.0`

# all:	mydrawwidget.so

# gtkmydrawwidget.o:	brushsettings.h gtkmydrawwidget.c gtkmydrawwidget.h
# 	cc $(CFLAGS) -c -o $@ gtkmydrawwidget.c

# gtkmybrush.o:	brushsettings.h gtkmybrush.c gtkmybrush.h
# 	cc $(CFLAGS) -c -o $@ gtkmybrush.c

# clean:
# 	rm *.o *.so brushsettings.h mydrawwidget.defs mydrawwidget.defs.c

# mydrawwidget.defs.c: mydrawwidget.defs mydrawwidget.override
# 	pygtk-codegen-2.0 --prefix mydrawwidget \
# 	--register $(DEFSDIR)/gdk-types.defs \
# 	--register $(DEFSDIR)/gtk-types.defs \
# 	--override mydrawwidget.override \
# 	mydrawwidget.defs > mydrawwidget.defs.c

# mydrawwidget.defs: gtkmydrawwidget.h gtkmybrush.h surface.h Makefile
# 	/usr/share/pygtk/2.0/codegen/h2def.py gtkmydrawwidget.h gtkmybrush.h > mydrawwidget.defs
# 	./caller_owns_return.py mydrawwidget.defs get_nonwhite_as_pixbuf get_as_pixbuf

# mydrawwidget.so: mydrawwidget.defs.c mydrawwidgetmodule.c gtkmydrawwidget.o surface.o gtkmybrush.o brush_dab.o helpers.o
# 	$(CC) $(LDFLAGS) $(CFLAGS) -shared $^ -o $@
