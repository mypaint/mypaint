# How to Contribute to MyPaint

MyPaint encourages contributions from everybody,
no matter what their background or identity.
We will try to help new developers so that
they feel confident in contributing to the project.

Please note that MyPaint is released with a
[Contributor Code of Conduct](CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.

There are many ways you can help make MyPaint
a really good program for everybody to use,
not limited to just writing program code.
Small tasks which are particularly suited for new contributors
get tagged as "bitesize" in our issue tracker:
look out for that label in the links below.

-----

## How to find us

We are on Twitter at [@MyPaintApp](https://twitter.com/MyPaintApp),
and many of us watch mentions of the word "mypaint".
Ask us questions, and we'll try to help out!

For more direct real-time conversions,
there is a MyPaint channel on IRC,
[#mypaint on FreeNode](irc://freenode/mypaint).

There's a mailing list too:
[mypaint-discuss](https://mail.gna.org/listinfo/mypaint-discuss).
All new contributors are especially encouraged to subscribe to it.

-----

## Test the program, and report issues

Report bugs in MyPaint on github,
at https://github.com/mypaint/mypaint/issues

* When reporting your issue,
  be sure to describe what operating system you're on
  and which version of MyPaint you're using.
* Please limit your issue reports to a single, fixable problem.
* If you are requesting a new feature,
  please submit mockups, pictures,
  or other clear explanations of what it is you want.
  And please tell us how the feature would help you.

The tracker is not really intended for user support,
but we can sometimes help investigate your problems there.
Your problem report may turn into a workaround you can use,
and many reports have helped us uncover
a more fundamental bug in the program code.
However don't be discouraged
if the issue gets closed when there's nothing we can do about it
in the current phase of development.

Note that Mypaint 1.0.0 is a *really* old version,
so if possible you should upgrade to the latest version before reporting anything,
as it's probably been fixed by now.
If you are on Windows, much more up-to-date builds are available:

* https://github.com/mypaint/mypaint/releases
  (signed releases made by the maintainer)
* http://opensourcepack.blogspot.fr/2013/01/mypaint-and-pygi.html
  (a more established, independent build, better tested)

-----

## Help make program artwork and icons

MyPaint's icons are primarily SVG symbolic icons, rendered at 16x16 pixels.
For certain uses such as the toolbar, we ask for 24x24 pixel versions too.
Try to be pixel-exact when designing icons,
that way the icons will look good at the sizes we need.

There is an Inkscape template for icon designers,
and an extraction script in the `svg` folder.

The tag for artwork and icons we need in the MyPaint bug tracker is "artwork".
[List of artwork requests](https://github.com/mypaint/mypaint/issues?q=is%3Aopen+is%3Aissue+label%3Aartwork).

TODO: probably need a word or two here on contributing artwork other than icons.
We don't have any splash screens or screenshots in the codebase,
but a graphical, interactive tutorial
like Inkscape's
would be brilliant to have.

TODO: brush sets and paper textures probably count as artwork.
They have particularly stringently _open_ licensing requirements,
so should probably mention that here.

-----

## Translate the program's text

MyPaint is translated into 26 languages and scripts at the time of writing,
excluding variant spellings of of English.
If you have a little technical knowledge
and good skills in a written language other then English,
you can help us by translating the text that gets displayed on the screen.

We use GNU gettext for i18n,
and have separate instructions for translators in [po/README.md](po/README.md).

---------------------------------------

## Contribute code and fix bugs

Technically oriented people can help here.
We will try to test and accept Github pull requests we receive,
and that is in fact the best way of sending us patches.

The tag for confirmed bugs in the MyPaint bug tracker is "bug".
[List of open bugs](https://github.com/mypaint/mypaint/issues?q=is%3Aopen+is%3Aissue+label%3Abug).

See [DEBUGGING.md](DEBUGGING.md) for details of
how to debug MyPaint most effectively
or run the profiler.

### Python style

Follow [PEP-8](http://legacy.python.org/dev/peps/pep-0008/)
for the most part.
There's a [neat tool](https://github.com/jcrocholl/pep8)
you can use to automatically find PEP-8 errors,
so you should preferrably run it before committing,
just to see if you've introduced any new errors.

There's a few things PEP-8 doesn't cover, though,
so here's a few more guidelines to follow:

#### Try to avoid visual indentation

Visual indentation makes it hard to maintain things,
and also ends up making things really cramped.

    # (don't do this)
    x = run_function_with_really_long_name(argument1,
                                           argument2)

For functions that take a *lot* of arguments,
it's a good idea to do something like:

    # (this is better)
    x = wow_this_sure_is_a_pretty_complicated_function(
        arg1, arg2,
        really_long_argument,
        named_arg = "something",
    )

This is also recommended for long array/tuple/etc literals:

    x = [
        "something",
        "another thing",
        "etc",
    ]

#### Strings

We prefer `str.format` (and `unicode.format`)
over C-style `"%s"` interpolation.
It's much easier to read,
and far friendlier to our translators.

The exception to this rule is for
hardcoded status messages sent to the program's log output(s).
The standard `logging` module requires C-style formatting,
so please use `%r` there for interpolated strings.

We have a custom gettext module in `lib/`
which you can import for "C macro"-like translation functions.
For new code, please always provide contexts.
The context string can be a `str` literal,
and while we're using Python 2.7 at least,
it's visually helpful to write `unicode` literals for the source string.

    label.set_text(C_(
        # Context string:
        "some module: some dialog: label",
        # Source string ("msgid") to be translated:
        u"Long Unicode message to be translated. "
        u"Our source language is US English. "
        u"You can use formatting codes here, “{like_this}”."
    )).format(
        like_this = "just like this",
    )

They have to be used almost as if they were actual C macros,
meaning you must write the strings in the `C_()` call as literals.
However `intltool` knows about Python constructs and formatting codes.

Favour standard US English punctuation for new translated strings,
and use proper Unicode punctuation like `“…’–”` instead of `"...'-`.

The practicalities of quoting things like filenames for the user to see
mean that it's better to use double curly quotes
in preference to single quotes.
They're easier to see,
and the user is more likely to have filenames with apostrophes in them.
Also, please place punctuation outside the final quote.

#### Don't commit commented-out code

Commented-out code, also known as dead code,
has the potential to cause a lot of harm
as commented-out code quickly becomes out of date and misleading.
We have version control anyway, so people can just look at the commit log.

