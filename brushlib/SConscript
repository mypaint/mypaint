Import('env', 'python')

env.Execute(python + ' generate.py')
env.Clean('.', 'brushsettings.h')
env.Clean('.', Glob('*.pyc'))

module = env.SharedLibrary('mypaint-brushlib', Glob("*.c"))

Return('module')
