Version 2.0.1:
* libmypaint requirement bumped to >= 1.6.
* HSV/HCY brush dynamics in 2.x mode fixed.
* Undo/Redo stack size made configurable and more efficient.
* Undo/Redo stack default size increased to 40.
* Rounding error when saving files in 2.x mode fixed.
* Invalid l11n strings fixed.
* Translated strings added/updated for several languages.
* Dependency on intltool removed (still required for libmypaint).
* Variables in l11n strings validated on build (xml tags are not validated).
* libmypaint translations are now bound, fixing them in the appimages.
* OARS rating added to appdata (relevant for app-distribution software).
* A number of Py3 compatibility issues were fixed.
* Svg icons fixed - some were previously considered invalid by librsvg >= 2.48.
* Svg icon sizes reduced across the board.
* Size of Windows install bundle significantly reduced.
* Typos in documentation fixed.

Version 2.0.0:
* Full Python3 support
* Tons of style consistency updates (flake8).
* Switch to PyGI
* New ability to simplify nodes in Inking tool.
* New ability to delete autosaves from recovery dialog.
* Improved user messages.
* Scratchpad no longer scrolls on size changes
* Integrated bug reporting w. issue template
* Use external libmypaint (>= 1.5).
* Most (possibly all) IO uses context managers.
* Layer manipulability dependent on visibility.
* Brush description shown in tooltip in quick choice dialog.
* AppImage builds added.
* Move build to setuptools/distutils.
* New keyboard shortcuts/defaults.
* New symmetry modes (vertical, vert+horz, rotational, snowflake).
* Flatpak support added (currently not functional?).
* Maximum input mapping curve points increased to 64.
* Curve editor points snaps on 0.5 increments.
* Input curve reset button clears the graph.
* New Import Layers feature.
* Zoom and view-rotation-dependent painting.
* New debug command: Vacuum Empty Tiles.
* Progress feedback for loading/saving.
* New Layer Properties dialog.
* New separate H/C/Y pickers.
* Per-document settings.
* New Layer Views.
* Layer rendering code reworked.
* New Remove Layer Backdrop command.
* New Refactor Layer Group commands.
* Linear compositing.
* Spectral mixing (pigment mode).
* Ability to emulate inputs with modifier keys.
* Support for barrel rotation input.
* Floodfill performance improved.
* Offset, feathering and gap detection added to Floodfill.
* Source, blend mode and opacity added to Floodfill.
* Erase / Lock Alpha / Colorize modes respected by Floodfill.
* Floodfill is now cancellable.
* Configurable layer blinking on layer switching.
* Fix crash when moving empty layer group.
* Interface language configurable in user preferences.
* Add version info to .ora files and compatibility checks.
* Bundled backgrounds optimized for size.
* Lots of bugs, crashers and other annoyances fixed.

Version 1.2.1:
* Fix failure to start under GLib 2.48.
* Fix failure to start when config and user data dirs are missing.
* GNOME: Update mypaint.appdata.xml.
* Fix failure to start when no translations are present.
* Fix pure-black being duplicated in the colour history.
* Fix glitch stroke when Inking is exited & the default tool entered.
* OSX: fix exception if AppKit isn't installed.
* Fix mispositioned windows in multi-monitor setups.
* Windows: fix inability to paste more than the 1st copied image.
* Fix exception when pasting into a layer group.
* Fix incorrect numeric range check on x-axis tilts.
* Fix layers blinking when selected in layer-solo mode.
* Fix palette drag issues with GTK 3.19.8+.
* Fix exception in the colours tab of the background chooser dialog.
* Fix UI glitch: mark cloned layer as selected after duplicate.
* Fix a potential exception with the brush and colour history features.
* About box: report versions better un Windows.
* Make sure layer clones get selected immediately.
* Fix hypersensitive tab drags.
* Fix allelerator mapping sort order.
* Fix exceptions when loading a corrupt thumbnail during thumb updates.
* Fix GTK removing the main canvas widget.
* BrushManager: use UUIDs for device brush names, backwards-compatibly.
* Fix repeated pixbuflist redraws.
* Windows: drop support for floating windows till upstream support's OK.
* Wayland: remove references to cursors that don't exist.

Version 1.2.0:
* New intuitive Inking tool for smooth strokes.
* New Flood Fill tool.
* Automated backups of your working docs, with recovery on startup.
* Improved symmetry-setting and frame-setting modes.
* New workspace UI: two sidebars, with dockable tabbed panels.
* Smoother scrolling and panning.
* New brush pack.
* New brush and color history panel.
* New layer trimming command in frame options.
* Added layer groups.
* New layer modes: several masking modes added.
* Add display filters: greyscale, simulate dichromacy for trichromats.
* New color wheel options: Red/Yellow/Blue, Red-Green/Blue-Yellow.
* Uses dark theme variant by default.
* Clearer icons, prettier freehand cursors.
* Device prefs allow glitchy devices to be restricted.
* Eraser mode no longer changes the size of the brush.
* New vector layers, editable in an external app (Inkscape recommended).
* New fallback layer types: non-PNG image, data.
* More kinds of images now work as backgrounds.
* Improved Windows support
* Ported to GTK3.
* Accelerator map editor has moved to preferences.
* Many other bugfixes, translations, and code quality improvements.

Version 1.1.0:
* geometry tools: sequence of lines, curved lines, ellipses
* new brush and layer blending modes; different layer merging
* new color dialog: palette and gamut mapping
* improved document frame, can be resized on canvas
* symmetric drawing mode
* old color changer ("washed") from 0.6 is available again
* toolbar improvements, e.g. move layer mode, pan/zoom
* revised cursor and on-canvas color picker
* better separation of mypaint's brush library; json brush file format
* translations, performance improvements, bugfixes, and more

Version 1.0.0:
* toolbar with color, brush selector and brush settings dropdown
* tool windows can be docked instead of floating
* locked alpha brush mode
* basic layer compositing modes
* new scratchpad area
* lots of other improvements (about 500 commits)

Version 0.9.1:
* several fixes for non-ascii file names, directories, layer names
* workaround for tablets reporting invalid tilt values
* rotation: fix direction while mirrored, change steps to 22.5 degrees
* store freedesktop thumbnails also when saving (for preview in other apps)
* reduce the minimal cursor size
* brush selector: remember state of the expander at the bottom
* fix glitch when changing the brush/color with a different input device
* osx: fix compile error
* windows: use AppData folder for settings
* some other minor fixes

Version 0.9.0:
* brush collection: updated better and smaller collection
* brushset import and export
* improvement for jaggy lines on Windows (might also fix saving problems)
* fixes for non-ascii brush- and filenames (for Windows mainly)
* sharper image for some zoom levels
* stylus tilt support
* persistence of selected brush and group
* file preview in open dialog
* configurable default save format and zoom level
* optimizations: faster startup, much faster saving
* lots of small improvements, bug fixes, optimizations
* updated translations: hu,es,ru,sv,nb,nn_NO,sl,ko,it
* improved exception dialog
* added GIMP-style subwindow toggle
* added GIMP-style cursor-menu
* usability improvements for brush selector and brush settings dialog
* added file->export action

Version 0.8.2:
* fix regression in 0.8.1 causing temporary layers to stick
* complain about unsupported pygtk version

Version 0.8.1:
* fixed memory leak: layer data was never freed, eg. when opening a new image
* fixed loading of layer names
* respect layer visibility when saving to PNG
* fixed a freeze in the exception dialog
* added empty "favorites" brushgroup
* Korean translation
* some minor gui fixes

Version 0.8.0:
* many new brushes contributed by various artists
* brushes organized into groups
* straight lines are possible (hold shift)
* basic layer dialog
* select brush from a stroke on the canvas
* improved color picker, show color while picking
* tools stay at top, only one taskbar entry (depending on your wm)
* faster zoomed-out view (30x speedup in some cases)
* i18n support added, translations in several languages
* new and revised color selectors
* big background patterns are possible (with limitations)
* can save all layers as numbered PNGs
* some drag&drop support
* many other minor enhancements and bugfixes
	
Version 0.7.1:
* bugfixes for win32 build
* limit the cursor size (problem on Windows, and X11 with Compiz)
* fixed brushes that lead to save problems (Windows only?)
* show filename in titlebar
* zoom on scrollwheel
* new brushes: splatter and marker
* other minor fixes

Version 0.7.0:
* color history popup
* merge layer down
* layer solo
* color changer can operate clickless (hold key down, release key to select)
* can save flattened transparent PNG
* recognize eraser end of the stylus
* elliptical dabs are possible now (aspect ratio)
* new brushes and background patterns
* save/load improved
  * fixed bugs that caused overwrite without asking
  * made OpenRaster the default file format
  * made "save scrap" more consistent (always save to scrap directory)
  * faster saving and loading (about factor two)
  * do dithering when converting from 16bit to 8bit (only when saving with transparency)
* fixed build problem with some distributions
* many other GUI tweaks and bugfixes

Version 0.6.0:
* Layers, transparency and eraser mode
* Background color and pattterns
* Save OpenRaster
* Very fast undo, limited number of steps
* Canvas rotation (via keyboard shortcuts) and mirroring
* A few great new brushes

Version 0.5.1:
* Fullscreen mode was implemented.
* Streamlined "Save" and "Save As". Also added error handling.
* Added "Save Next" as a dialog-free non-destructive alternative.
* If undo would take a lot of time, show a confirmation dialog.
* Reduced maximum zoom-out to 1/4 to avoid out of memory.
* Added settings dialog with global pressure mapping.
* New desktop icon by Sebastian Kraft (needcoffee).
* Fixed "ghost strokes" seen on internet tablets.
* Fixed startup crash where the mouse was wrongly recognized as a tablet.
* Fixed compilation for some systems.
* New brushes were added (and some removed).

Before 0.5.0:
* no changelog available

