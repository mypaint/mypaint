import os
SConsignFile() # no .scsonsign into $PREFIX please

opts = Options('options.cache', ARGUMENTS)
opts.Add(PathOption('PREFIX', 'Directory to install under', '/usr/local'))
env = Environment(ENV=os.environ, options=opts)
opts.Update(env)
opts.Save('options.cache', env)

env.ParseConfig('python-config --cflags --ldflags')

# tilelib
env2 = env.Clone()
env2.Append(SWIGFLAGS='-python -noproxy')
sources = '''
tilelib/ctile.i
'''
tilelib_module = env2.LoadableModule('tilelib/ctile.so', Split(sources), LDMODULEPREFIX='')


# mypaint application

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
module = env.LoadableModule('mydrawwidget', Split(sources), LDMODULEPREFIX='')


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

# stub (this file would get generated with autotools)
env.Command('config.h', [], 'echo > $TARGET')



# installation

#env.Install(module, '$PREFIX/lib/mypaint') # location for private compiled extensions
##env.Install(module, '$PREFIX/share/mypaint') # theoretical location for private pure python modules (meld uses $PREFIX/lib/meld)
#env.Install(data, '$PREFIX/share/mypaint')
