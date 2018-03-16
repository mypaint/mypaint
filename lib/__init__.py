
from __future__ import division, print_function


def myfunc(a):
    """This newly-added function contains several flake8 style violations. In particular, E501, E225, and F821 (twice, or once, depending), E265, and W291. It's only here for testing out Hound-CI."""
    #You shouldn't write code looking like this.   
    a+=b
    return unicode(a)
