%module ctile
%{
#include "ctile.c"
%}

%include "ctile.c"

%init %{
import_array();
%}
