import os

env = Environment(ENV = os.environ)
env.ParseConfig('python-config --cflags --ldflags')
env.ParseConfig('pkg-config --cflags --libs gtk+-2.0 pygtk-2.0')

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

# the main python module
# see also http://www.scons.org/wiki/PythonExtensions
env.LoadableModule('mydrawwidget', Split(sources), LDMODULEPREFIX='')


# code generators

env.Command('brushsettings.h', ['generate.py', 'brushsettings.py'], './generate.py')

defsdir = env.backtick('pkg-config pygtk-2.0 --variable=defsdir').strip()

pygtk_codegen = 'pygtk-codegen-2.0'
pygtk_h2def = '/usr/share/pygtk/2.0/codegen/h2def.py'
glib_genmarshal = 'glib-genmarshal'

env.Command('mydrawwidget.defs.c', ['mydrawwidget.defs', 'mydrawwidget.override'], '''
%s --prefix mydrawwidget \
--register %s/gdk-types.defs \
--register %s/gtk-types.defs \
--override mydrawwidget.override \
mydrawwidget.defs > $TARGET
''' % (pygtk_codegen, defsdir, defsdir))

defs_inputs = 'gtkmydrawwidget.h gtkmybrush.h gtkmysurface.h gtkmysurfaceold.h'
env.Command('mydrawwidget.defs', Split(defs_inputs + ' fix_generated_defs.py'), '''
python %s %s > mydrawwidget.defs && ./fix_generated_defs.py
''' % (pygtk_h2def, defs_inputs))

env.Command('mymarshal.h', 'mymarshal.list', glib_genmarshal + ' --prefix=mymarshal --header $SOURCE > $TARGET')
env.Command('mymarshal.c', 'mymarshal.list', glib_genmarshal + ' --prefix=mymarshal --body   $SOURCE > $TARGET')

