# How to Contribute to MyPaint

MyPaint encourages contributions from everybody, no matter what their background or identity. We will try to help new developers so that they feel confident in contributing to the project.

There are many ways you can help make MyPaint a really good program for everybody to use, not limited to just writing program code. Small tasks which are particularly suited for new contributors get tagged as "bitesize" in our issue tracker: look out for that label in the links below.

-----

## How to find us

We are on Twitter at [@MyPaintApp](https://twitter.com/MyPaintApp), and many of us watch mentions of the word "mypaint". Ask us questions, and we'll try to help out!

For more direct real-time conversions, there is a MyPaint channel on IRC, [#mypaint on FreeNode](irc://freenode/mypaint).

There's a mailing list too: [mypaint-discuss](https://mail.gna.org/listinfo/mypaint-discuss). All new contributors are especially encouraged to subscribe to it.

-----

## Test the program, and report issues

Report bugs in MyPaint on github, at https://github.com/mypaint/mypaint/issues

* When reporting your issue, be sure to describe what operating system you're on and which version of MyPaint you're using.
* Please limit your issue reports to a single, fixable problem.
* If you are requesting a new feature, please submit mockups, pictures or other clear explanations of what it is you want, and how it would help you.

The tracker is not really intended for user support, but we can sometimes help investigate your problems there. Your problem report may turn into a workaround you can use, and many reports have helped us uncover a more fundamental bug in the program code. However don't be discouraged if the issue gets closed when there's nothing we can do about it in the current phase of development.

Note that Mypaint 1.0.0 is a *really* old version, so if possible you should upgrade to the latest version before reporting anything, as it's probably been fixed by now. If you are on Windows, look here for relatively up-to-date builds: http://opensourcepack.blogspot.fr/2013/01/mypaint-and-pygi.html

-----

## Help make program artwork and icons

MyPaint's icons are primarily SVG symbolic icons, rendered at 16x16 pixels. For certain uses such as the toolbar, we ask for 24x24 pixel versions too. Try to be pixel-exact when designing icons, that way the icons will look good at the sizes we need.

There is an Inkscape template for icon designers, and an extraction script in the `svg` folder.

You may see some remnants of an earlier, "Tango" style. Please don't contribute new icons in this style, it is due to be phased out.

The tag for artwork and icons we need in the MyPaint bug tracker is "artwork". [List of artwork requests](https://github.com/mypaint/mypaint/issues?q=is%3Aopen+is%3Aissue+label%3Aartwork).

TODO: probably need a word or two here on contributing artwork other than icons. We don't have any splash screens or screenshots in the codebase, but a graphical, interactive tutorial like Inkscape's would be brilliant to have.

TODO: brush sets and paper textures probably count as artwork. They have particularly stringently _open_ licensing requirements, so should probably mention that here.

-----

## Translate the program's text

MyPaint is translated into 20 languages and scripts at the time of writing, excluding variant spellings of of English. If you have a little technical knowledge and good skills in a written language other then English, you can help us by translating the text that gets displayed on the screen.

We use GNU gettext for i18n, and have separate instructions for translators in [po/README.md](po/README.md).

---------------------------------------

## Contribute code and fix bugs

Technically oriented people can help here. We will try to test and accept Github pull requests we receive, and that is in fact the best way of sending us patches.

The tag for confirmed bugs in the MyPaint bug tracker is "bug". [List of open bugs](https://github.com/mypaint/mypaint/issues?q=is%3Aopen+is%3Aissue+label%3Abug).

### Debugging

By default, our use of Python's ``logging`` module
is noisy about errors, warnings, and general informational stuff,
but silent about anything with a lower priority.
To see all messages, set the ``MYPAINT_DEBUG`` environment variable.

  ```sh
  MYPAINT_DEBUG=1 ./mypaint -c /tmp/cfgtmp_throwaway_1
  ```

MyPaint normally logs Python exception backtraces to the terminal
and to a dialog within the application.

To debug segfaults in C/C++ code, use ``gdb`` with a debug build,
after first making sure you have debugging symbols for Python and GTK3.

  ```sh
  sudo apt-get install gdb python2.7-dbg libgtk-3-0-dbg
  scons debug=1
  export MYPAINT_DEBUG=1
  gdb -ex r --args python ./mypaint -c /tmp/cfgtmp_throwaway_2
  ```

Execute ``bt`` within the gdb environment for a full backtrace.
See also: https://wiki.python.org/moin/DebuggingWithGdb

### Python style

Follow [PEP-8](http://legacy.python.org/dev/peps/pep-0008/) for the most part. There's a [neat tool](https://github.com/jcrocholl/pep8) you can use to automatically find PEP-8 errors, so you should preferrably run it before committing, just to see if you've introduced any new errors.

There's a few things PEP-8 doesn't cover, though, so here's a few more guidelines to follow:

#### Try to avoid visual indentation

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

#### Don't commit commented-out code

Commented-out code, also known as dead code, has the potential to cause a lot of harm as commented-out code quickly becomes out of date and misleading. We have version control, anyway, so people can just look at the commit log.

