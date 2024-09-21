# Translating mypaint

[![Translation status](https://hosted.weblate.org/widgets/mypaint/-/svg-badge.svg)](https://hosted.weblate.org/engage/mypaint/?utm_source=widget)

MyPaint uses WebLate for its translations.
You can get involved at <https://hosted.weblate.org/engage/mypaint/>.

That might be all the information you need
if you just want to help ensure that MyPaint
is correctly translated into your language.

# Information for coders and maintainers

We use [GNU gettext][gettext] for runtime translation of program text.

Please note that there are two projects that need translating for the
MyPaint app:

* libmypaint, the brush engine library that MyPaint uses.
* MyPaint itself.

both projects are exposed on WebLate.

The rest of this document covers translating
the MyPaint application itself, minus libmypaint.

## Making strings translatable

This section gives an overview of how to make strings translatable
in the source code. For in-depth technical details of the workings,
of gettext, refer to [its manual][gettext].

Strings that need to be translatable have to be used as arguments in calls
to particular translation functions, also referred to as "macros"
(preprocessor macros is a standard way of using gettext in C, the python calls
used are not actually macros, but rather functions or function aliases).
The purpose of these calls are twofold: they make it possible to extract the
translatable strings and related information from the source code _and_ they
perform the actual translation lookups at runtime.

> **Don't use aliases of the translation functions!**
> Even though the _lookup_ will still work, the _extraction_
> process is based on keywords, and cannot deduce aliasing, resulting
> in the strings not being extracted to the translation template.

### Basic translation: `_(message)`

The `_` function is an alias of the standard gettext function, and takes
the string _literal_ of the message to be translated as its only argument.
If there is a comment block immediately preceding the **first line**
of the _argument_, that comment will be extracted as well to provide
information about the string to translators.

When extracted, the argument string will appear as a msgid field in the translation
files, in conjunction with a set of extracted comments and source code locations.

Most of the time the `C_` function should be favored over `_`, when creating new strings,
but be careful about replacing existing `_(...)` calls with `C_(...)`.
See the following sections for details.

### Context-specific translation: `C_(context, message)`

The `C_` "macro" takes a context string literal in addition to the message
string literal. The context has two purposes: it can provide useful
information to the translators about how the string is used and it makes
it _possible_ to have different translations for the same message string
(when an identical message literal is used in different places).

For more information, see [the relevant section in the gettext manual][gettext_ctx]

Be careful about changing existing calls to `_` with calls to `C_`
just to add contextual information when there is no need for disambiguation.
The reason is that such changes will mark existing translations of that message
as fuzzy - and such translations will not be used until the fuzzy flag has been
removed (usually by a translator reviewing changes).
If you just want to add documentation, use a translator comment.

### Adding translator comments

An important thing to know about _translator comments_ is that they are only
extracted when preceding the line that the string literal of the _message_
**starts on**. See the examples below for how comment placement matters.

<details>
<summary>Examples</summary>

```
# This comment will be extracted
translated = _("my message")

# This comment will be extracted too
translated = C_("context of my message", "my message")

# Both of these comment lines will
# be extracted; they form a block.
translated = C_("context of my message", "my message")

# This comment will NOT be extracted!
translated = _(
    "this message is really long, and has been placed on its very own line"
)

translated = _(
    # But this comment WILL be extracted
    "this message is really long, and has been placed on its very own line"
)

translated = C_(
    # This comment precedes the CONTEXT line and will NOT be extracted!
    "context of my message",
    "my message"
)

translated = C_(
    "context of my message",
    # This comment precedes the MESSAGE line and WILL be extracted
    "my message"
)

```
</details>

### Using variables in message strings

When the translated strings contain variables, use python brace format:
```
translated = C_(
    "shmoo-counter: status message",
	# TRANSLATORS: We have to use pounds for reasons of historical accuracy
	"Counted {num_shmoos} shmoos, weighing a total of {total_weight} pounds."
	).format(num_shmoos=len(shmoos), total_weight=sum([s.weight for s in shmoos]))
```

_Never_ stitch together translated fragments with variables - it will always
make translation very difficult. For many languages doing it  may even make it
impossible to produce translations with correct grammar, due to the order of
arguments being impossible to change.

## After updating program strings

After adding or changing any string in the source text which
makes use of the gettext macros, you will need to manually run

    cd po/
    ./update_potfiles.sh

    ./update_languages.sh

and then commit the modified `po/mypaint.pot` & `po/*.po` too,
along with your changes.

The `.pot` file alone can be updated by running
just the first command,
if all you want to do is compare diffs.

The `update_metadata.sh` script does not need to be run on a regular basis.
It is just a convenience for generating generic headers for new language files
and for updating the version field of existing language files.

# Information for translators

## New translation (manual)

Start by putting a new stub `.po` file into this directory.

To make such a file you can
copy the header from an existing `.po` file
and modify it accordingly.

Unless there are several country-specific dialects for your language,
the file should be named simply `ll.po`
where "ll" is a recognized [language code][ll].

If there are several dialects for your language,
the file should be named `ll_CC.po`
where "CC" is the [country code][CC].

Before you can work on it,
you will need to update the `.po` file
from the most recent `.pot` template file
generated by the developers.

## Update translation (manual)

Before working on a translation,
update the `po` file for your language.
For example, for the French translation, run:

    update_languages.sh fr.po

## Use/Test the translation

After modifying the translation,
you can run MyPaint in demo mode to see the changes:

    python setup.py demo

If you want to run MyPaint with a specific translation on Linux,
here are two ways of doing so:

You can either use the LANG environment variable
like this (the locale needs to be supported):

	LANG=ll_CC.utf8 python setup.py demo

where "ll" is a [language code][ll], and and "CC" is a [country code][CC].

If you don't have to locale of the language you want
to test installed, you can use the LANGUAGE variable
instead:

	LANGUAGE=ll_CC python setup.py demo

> You don't have to supply the country code (CC) if you
> don't need to disambiguate multiple dialects.

> When using `LANGUAGE` without having the locale installed,
> some strings that are not part of MyPaint will not be translated.
> For example, button strings in file chooser dialogs.

To run MyPaint with the original strings, for comparison,
you can use the `LC_MESSAGES` variable like this:

    LC_MESSAGES=C python setup.py demo

## Send changes (manual)

Before you send your changes, please make sure that
your changes are based on the
current development (git) version of MyPaint.

We prefer changes as [Github pull requests][PR],
but if you do not know git you can also [open a new issue on github][NEW_ISSUE]
and attach either a unified diff or the new/updated .po file to it.

## Weblate

Weblate provides a browser-based interface for adding, editing
and updating translations. It's a very good way to provide
translations without having to worry about the technical details.

If you are interested in keeping the translations up to date,
please subscribe to the MyPaint project on WebLate:
<https://hosted.weblate.org/accounts/profile/#subscriptions>.

-------------------

[gettext]: http://www.gnu.org/software/hello/manual/gettext/ (Official GNU gettext manual)
[gettext_ctx]: https://www.gnu.org/software/gettext/manual/gettext.html#Contexts (gettext manual section on message contexts)
[ll]: http://www.gnu.org/software/hello/manual/gettext/Usual-Language-Codes.html#Usual-Language-Codes ("ll" options)
[CC]: http://www.gnu.org/software/hello/manual/gettext/Country-Codes.html#Country-Codes ("CC" options)
[PR]: https://help.github.com/articles/using-pull-requests/
[NEW_ISSUE]: https://github.com/mypaint/mypaint/issues/new?title=Manual+translations+for:+LANGUAGE&body=New%2Fupdated+translations+for+%2E%2E%2E
