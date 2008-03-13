%module _tilelib
%{
#include "tilelib.h"
#include "Numeric/arrayobject.h"
%}

%include "tilelib.h"

%init %{
import_array();
%}
