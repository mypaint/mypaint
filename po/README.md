# Information for translators

We use [GNU gettext][gettext] for runtime translation of program text.

If you're translating an official release tarball,
the commands listed below should just work as written.

If you're working on the development master,
you first need to ensure that your source tree is complete.
We make use of submodules, so
if you just did:

    git clone https://github.com/mypaint/mypaint.git

then you also need to do:

    cd mypaint
    git submodule update --init --force

before running the commands below.

## New translation

To start a new language, generate only the template `mypaint.pot` with:

    scons translate=pot

Then put a new `.po` file into this directory.
To make such a file you can copy the header from an existing `.po` file
and modify it accordingly.

Unless there are several country-specific dialects for your language,
the file should be named simply `ll.po`
where "ll" is a recognized [language code][ll].

If there are several dialects for your language,
the file should be named `ll_CC.po`
where "CC" is the [country code][CC].

## Update translation

Before working on a translation, update the `po` file for your language.
For example, for the French translation, run:

    scons translate=fr

## Use/Test the translation

After modifying the translation you need to rebuild to see the changes:

    scons

To run MyPaint with a specific translation on Linux,
you can use the LANG environment variable
like this (the locale needs to be supported):

    LANG=ll_CC.utf8 ./mypaint

where "ll" is a [language code][ll], and and "CC" is a [country code][CC].
Your working directory must be the root directory of the mypaint source.

To run MyPaint with the original strings, for comparison,
you can use the `LC_MESSAGES` variable like this:

    LC_MESSAGES=C ./mypaint

## Send changes

Before you send your changes, please make sure that
your changes are based on the current development (git) version of MyPaint.

We prefer changes as [Github pull requests][PR],
but if you do not know git just send
either a unified diff or the updated .po file
along with your name to: *a.t.chadwick (AT) gmail.com*.

If you are interested in keeping the transalations up to date,
please subscribe to *mypaint-discuss (AT) gna.org*.

[gettext]: http://www.gnu.org/software/hello/manual/gettext/ (Official GNU gettext manual)
[ll]: http://www.gnu.org/software/hello/manual/gettext/Usual-Language-Codes.html#Usual-Language-Codes ("ll" options)
[CC]: http://www.gnu.org/software/hello/manual/gettext/Country-Codes.html#Country-Codes ("CC" options)
[PR]: https://help.github.com/articles/using-pull-requests/
