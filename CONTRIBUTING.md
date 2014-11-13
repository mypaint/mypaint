# Issue Reporting

Report bugs in MyPaint on github, at https://github.com/mypaint/mypaint/issues

When reporting your issue, be sure to describe what operating system you're on and which version of Mypaint you're using.

Please limit your issue reports to a single, fixable problem. If you are requesting a new feature, please submit mockups, pictures or other clear explanations of what it is you want, and how it would help you.

Note that Mypaint 1.0.0 is a *really* old version, so if possible you should upgrade to the latest version before reporting anything, as it's probably been fixed by now. If you are on Windows, look here for relatively up-to-date builds: http://opensourcepack.blogspot.fr/2013/01/mypaint-and-pygi.html

# Style Guide

## Python

Follow [PEP-8](http://legacy.python.org/dev/peps/pep-0008/) for the most part. There's a [neat tool](https://github.com/jcrocholl/pep8) you can use to automatically find PEP-8 errors, so you should preferrably run it before committing, just to see if you've introduced any new errors.

There's a few things PEP-8 doesn't cover, though, so here's a few more guidelines to follow:

### Try to avoid visual indentation

Visual indentation makes it hard to maintain things, and also ends up making things really cramped. When you would write:
```python
x = run_function_with_really_long_name(argument1,
                                       argument2)
```
Instead, do:
```python
x = run_function_with_really_long_name(
    argument1,
    argument2)
```

For functions that take a *lot* of arguments, it's a good idea to do something like:
```python
x = wow_this_sure_is_a_pretty_complicated_function(
    arg1, arg2,
    really_long_argument,
    named_arg="something",
)
```

This is also recommended for long array/tuple/etc literals:
```python
x = [
    "something",
    "another thing",
    "etc",
]
```

### Don't commit commented-out code

Commented-out code, also known as dead code, has the potential to cause a lot of harm as commented-out code quickly becomes out of date and misleading. We have version control, anyway, so people can just look at the commit log.

## C++


