#include <Python.h>

char* launchscript = "\
import sys;\
sys.path.insert(0,'python25.zip');\
sys.path.insert(0,'site-packages.zip');\
sys.path.insert(0,'shared');\
sys.argv.append('-c');\
sys.argv.append('config');\
execfile('mypaint');\
";

int main(int argc, char *argv[])
{
	Py_Initialize();
	PySys_SetArgv(argc, argv);
	PyRun_SimpleString(launchscript);
	Py_Finalize();
  return 0;
}

