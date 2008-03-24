%module mypaintlib
%{
#include "mypaintlib.hpp"
%}

typedef struct { int x, y, w, h; } Rect;

%include "tilelib.hpp"
%include "brush.hpp"

%init %{
import_array();
%}

