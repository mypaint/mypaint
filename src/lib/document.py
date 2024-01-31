# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2010-2019 by the MyPaint Development Team
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

from __future__ import absolute_import, division, print_function

import os
import sys
import zipfile
import tempfile
import time
from os.path import join
import xml.etree.ElementTree as ET
from warnings import warn
import shutil
from datetime import datetime
from collections import namedtuple
import json
import logging
from lib.fileutils import safename
from lib.naming import make_unique_name

from lib.gibindings import GObject
from lib.gibindings import GLib

import lib.meta
import lib.helpers as helpers
import lib.fileutils as fileutils
import lib.tiledsurface as tiledsurface
import lib.command as command
import lib.layer as layer
import lib.brush as brush
from lib.observable import event
from lib.observable import ObservableDict
import lib.pixbuf
from lib.cache import DEFAULT_CACHE_SIZE
from lib.errors import FileHandlingError
from lib.errors import AllocationError
import lib.idletask
from lib.gettext import C_
import lib.xml
import lib.glib
import lib.feedback
import lib.layervis
from lib.pycompat import unicode

logger = logging.getLogger(__name__)


## Module constants

DEFAULT_RESOLUTION = 72
DEFAULT_UNDO_STACK_SIZE = 40

N = tiledsurface.N

CACHE_APP_SUBDIR_NAME = u"mypaint"
CACHE_DOC_SUBDIR_PREFIX = u"doc."
CACHE_DOC_AUTOSAVE_SUBDIR = u"autosave"
CACHE_ACTIVITY_FILE = u"active"
CACHE_UPDATE_INTERVAL = 10  # seconds

# Logging and error reporting strings
_LOAD_FAILED_COMMON_TEMPLATE_LINE = C_(
    "Document IO: common error strings: {error_loading_common}",
    u"Error loading “{basename}”."
)
_ERROR_SEE_LOGS_LINE = C_(
    "Document IO: common error strings: {see_logs}",
    u"The logs may have more detail "
    u"about this error."
)

# OpenRaster dialect consts

_ORA_MYPAINT_VERSION \
    = "{%s}version" % (lib.xml.OPENRASTER_MYPAINT_NS,)

_ORA_FRAME_ACTIVE_ATTR \
    = "{%s}frame-active" % (lib.xml.OPENRASTER_MYPAINT_NS,)

_ORA_UNSAVED_PAINTING_TIME_ATTR \
    = "{%s}unsaved-painting-time" % (lib.xml.OPENRASTER_MYPAINT_NS,)

_ORA_JSON_SETTINGS_ATTR \
    = "{%s}json-settings" % (lib.xml.OPENRASTER_MYPAINT_NS,)

_ORA_EOTF_ATTR \
    = "{%s}eotf" % (lib.xml.OPENRASTER_MYPAINT_NS,)


_ORA_JSON_SETTINGS_ZIP_PATH = "data/mypaint-settings.json"

## Class defs


_AUTOSAVE_INFO_FIELDS = (
    "path",
    "last_modified",
    "thumbnail",
    "num_layers",
    "unsaved_painting_time",
    "width", "height",
    "valid",
    "cache_in_use",
)


class AutosaveInfo (namedtuple("AutosaveInfo", _AUTOSAVE_INFO_FIELDS)):
    """Information about an autosave dir.

    :ivar unicode path: Full path to the autosave directory itself
    :ivar datetime.datetime last_modified: When its data was last changed
    :ivar GdkPixbuf.Pixbuf thumbnail: 256x256 pixel thumbnail, or None
    :ivar int num_layers: how many data layers exist in the doc
    :ivar float unsaved_painting_time: seconds of unsaved painting
    :ivar int width: Width of the document
    :ivar int height: Height of the document
    :ivar bool valid: True if the directory looks structurally valid
    :ivar bool cache_in_use: True if the directory is possibly in use

    """

    @classmethod
    def new_for_path(cls, path):
        if not isinstance(path, unicode):
            raise ValueError("path argument must be unicode")
        if not os.path.isdir(path):
            raise ValueError("Autosave folder %r does not exist", path)
        has_data = os.path.isdir(os.path.join(path, "data"))
        valid = has_data
        stackxml_path = os.path.join(path, "stack.xml")
        thumbnail_path = os.path.join(path, "Thumbnails", "thumbnail.png")
        has_stackxml = os.path.isfile(stackxml_path)
        if has_stackxml:
            last_modified = _get_path_mtime(stackxml_path)
        else:
            last_modified = _get_path_mtime(path)
            valid = False
        unsaved_painting_time = 0
        num_layers = 0
        width = 0
        height = 0
        if has_stackxml:
            try:
                doc = ET.parse(stackxml_path)
            except (ET.ParseError, IOError, OSError) as ex:
                valid = False
            else:
                image_elem = doc.getroot()
                unsaved_painting_time = max(0.0, float(
                    image_elem.attrib.get(
                        "mypaint_unsaved_painting_time",
                        0.0,
                    )
                ))
                width = max(0, int(image_elem.attrib.get('w', 0)))
                height = max(0, int(image_elem.attrib.get('h', 0)))
                num_layers = max(0, len(image_elem.findall(".//layer")) - 1)
        thumbnail = None
        if os.path.exists(thumbnail_path):
            thumbnail = lib.pixbuf.load_from_file(thumbnail_path)
        cache_in_use = False
        cache_dir_path = os.path.dirname(path)
        activity_file_path = os.path.join(cache_dir_path, CACHE_ACTIVITY_FILE)
        if os.path.exists(activity_file_path):
            cache_activity_time = _get_path_mtime(activity_file_path)
            cache_activity_dt = (datetime.now() - cache_activity_time).seconds
            if cache_activity_dt <= CACHE_UPDATE_INTERVAL + 3:
                cache_in_use = True
        return cls(
            path = path,
            last_modified = last_modified,
            valid = valid,
            num_layers = num_layers,
            unsaved_painting_time = unsaved_painting_time,
            width = width,
            height = height,
            thumbnail = thumbnail,
            cache_in_use = cache_in_use,
        )

    def get_description(self):
        """Human-readable description of the autosave

        :rtype: unicode

        """
        fmt_time = lib.helpers.fmt_time_period_abbr
        unsaved_time_str = fmt_time(self.unsaved_painting_time)
        last_modif_dt = (datetime.now() - self.last_modified).seconds
        # TODO: Look into whether breaking out these strings causes problems
        if last_modif_dt < 0:
            last_modif_ago_str = C_(
                "Document autosave descriptions: the {ago} string: future",
                # TRANSLATORS: string used in e.g. "3h42m from now"
                u"from now"
            )
        else:
            last_modif_ago_str = C_(
                "Document autosave descriptions: the {ago} string: past",
                # TRANSLATORS: string used in e.g. "8s ago"
                u"ago"
            )
        last_modif_str = fmt_time(abs(last_modif_dt))
        if self.cache_in_use:
            template = C_(
                "Document autosave descriptions",
                # TRANSLATORS: String descriptions for an autosaved backup.
                # TRANSLATORS: Time strings are localized: e.g. "3h42m" or "8s"
                u"Cache folder still may be in use.\n"
                u"Are you running more than once instance of MyPaint?\n"
                u"Close app and wait {cache_update_interval}s to retry."
            )
        elif not self.valid:
            template = C_(
                "Document autosave descriptions",
                u"Incomplete backup updated {last_modified_time} {ago}"
            )
        else:
            template = C_(
                "Document autosave descriptions",
                # TRANSLATORS: The {ago} variable is a translated string.
                # TRANSLATORS: Look for the msgids "from now" and "ago"
                # TRANSLATORS: (with context starting w. "Document autosave")
                # TRANSLATORS: to make sure their translations match this one.
                u"Backup updated {last_modified_time} {ago}\n"
                u"Size: {autosave.width}×{autosave.height} pixels, "
                u"Layers: {autosave.num_layers}\n"
                u"Contains {unsaved_time} of unsaved painting."
            )
        return template.format(
            autosave = self,
            unsaved_time = unsaved_time_str,
            last_modified_time = last_modif_str,
            ago = last_modif_ago_str,
            cache_update_interval = CACHE_UPDATE_INTERVAL,
        )


class Document (object):
    """In-memory representation of everything to be worked on & saved

    This is the "model" in the Model-View-Controller design for the
    drawing canvas. The View mostly resides in `gui.tileddrawwidget`,
    and the Controller is mostly in `gui.document` and `gui.mode`.

    The model contains everything that the user would want to save. It
    is possible to use the model without any GUI attached (see
    ``../tests/``).

    Please note the following difficulty with the command stack: many
    uses of the working Document model rely on altering the model
    directly, then writing an undoable record of the changes to the
    command stack when asked. There may be more than one concurrent
    source of these pending changes. See `sync_pending_changes()` for
    details of how sources of pending changes are asked to synchronize
    their state with the model and its command stack.

    """

    ## Class constants

    #: Debugging toggle. If True, New and Load and Remove Layer will create a
    #: new blank painting layer if they empty out the document.
    CREATE_PAINTING_LAYER_IF_EMPTY = True

    ## Initialization and cleanup

    def __init__(self, brushinfo=None, painting_only=False,
                 cache_dir=None, cache_size=DEFAULT_CACHE_SIZE,
                 max_undo_stack_size=DEFAULT_UNDO_STACK_SIZE):
        """Initialize

        :param brushinfo: the lib.brush.BrushInfo instance to use
        :param painting_only: only use painting layers
        :param cache_dir: use an existing cache dir
        :param cache_size: size of the layer render cache

        If painting_only is true, then no tempdir will be created by the
        document when it is initialized or cleared.

        If an existing cache dir is requested, it will not be created, and
        it won't be managed. Autosave and cleanup will be turned off if
        this is set; it's assumed that you're importing into a parent
        document.

        """
        object.__init__(self)
        if not brushinfo:
            brushinfo = brush.BrushInfo()
            brushinfo.load_defaults()
        self._layers = layer.RootLayerStack(self, cache_size=cache_size)
        self._layers.layer_content_changed += self._canvas_modified_cb
        self.brush = brush.Brush(brushinfo)
        self.brush.brushinfo.observers.append(self.brushsettings_changed_cb)
        self.stroke = None
        self.command_stack = command.CommandStack(max_undo_stack_size)

        # Cache and auto-saving to the cache
        self._painting_only = painting_only
        self._cache_dir = cache_dir
        if cache_dir is not None:
            if painting_only:
                raise ValueError(
                    "painting_only and cache_dir arguments "
                    "are mutually exclusive",
                )
            if not os.path.isdir(cache_dir):
                raise ValueError(
                    "cache_dir argument must be the path "
                    "to an existing directory",
                )
            self._owns_cache_dir = False
        else:
            self._owns_cache_dir = True
        self._cache_updater_id = None
        self._autosave_backups = False
        self.autosave_interval = 10
        self._autosave_processor = None
        self._autosave_countdown_id = None
        self._autosave_dirty = False
        if (not painting_only) and self._owns_cache_dir:
            self._autosave_processor = lib.idletask.Processor()
            self.command_stack.stack_updated += self._command_stack_updated_cb
            self.effective_bbox_changed += self._effective_bbox_changed_cb

        # Optional page area and resolution information
        self._frame = [0, 0, 0, 0]
        self._frame_enabled = False
        self._xres = None
        self._yres = None

        #: Document-specific settings, serialized as JSON when saving ORA.
        self._settings = ObservableDict()
        self.sync_pending_changes += self._settings_sync_pending_changes_cb

        #: Sets of layer-views, identified by name.
        self._layer_view_manager = lib.layervis.LayerViewManager(self)

        # And begin in a known state
        self.clear()

    def __repr__(self):
        bbox = self.get_bbox()
        nlayers = len(list(self.layer_stack.walk()))
        return ("<Document nlayers=%d bbox=%r paintonly=%r>" %
                (nlayers, bbox, self._painting_only))

    ## Layer stack access

    @property
    def layer_stack(self):
        """The root of the layer stack tree

        See also `lib.layer.RootLayerStack`.
        """
        # TODO: rename or alias this to just "layers" one day.
        return self._layers

    ## Working document's cache directory

    @property
    def tempdir(self):
        """The working doc's cache dir (read-only, deprecated name)"""
        warn("Use cache_dir instead", DeprecationWarning, stacklevel=2)
        return self._cache_dir

    @property
    def cache_dir(self):
        """The working document's cache dir"""
        return self._cache_dir

    def _create_cache_dir(self):
        """Internal: creates the working-document cache dir if needed."""
        if self._painting_only or not self._owns_cache_dir:
            return
        assert self._cache_dir is None
        app_cache_root = get_app_cache_root()
        doc_cache_dir = tempfile.mkdtemp(
            prefix=CACHE_DOC_SUBDIR_PREFIX,
            dir=app_cache_root,
        )
        if not isinstance(doc_cache_dir, unicode):
            doc_cache_dir = doc_cache_dir.decode(sys.getfilesystemencoding())
        logger.debug("Created working-doc cache dir %r", doc_cache_dir)
        self._cache_dir = doc_cache_dir
        # Start the cache updater, which kicks off background autosaves,
        # and updates an activity canary file.
        # Not a perfect solution, but maybe a better cross-platform one
        # than file locking, pidfiles or other horrors.
        activity_file_path = os.path.join(doc_cache_dir, CACHE_ACTIVITY_FILE)
        with open(activity_file_path, "w") as fp:
            fp.write(
                "A recent timestamp on this file indicates that\n"
                "its containing cache subfolder is active.\n"
            )
        self._start_cache_updater()

    def _cleanup_cache_dir(self):
        """Internal: recursively delete the working-document cache_dir if OK.

        Also stops any background tasks which update it.

        """
        if self._painting_only or not self._owns_cache_dir:
            return
        if self._cache_dir is None:
            return
        self._stop_cache_updater()
        self._stop_autosave_writes()
        shutil.rmtree(self._cache_dir, ignore_errors=True)
        if os.path.exists(self._cache_dir):
            logger.error(
                "Failed to remove working-doc cache dir %r",
                self._cache_dir,
            )
        else:
            logger.debug(
                "Successfully removed working-doc cache dir %r",
                self._cache_dir,
            )
        self._cache_dir = None

    def cleanup(self):
        """Cleans up any persistent state belonging to the document.

        This method is called by the main app's exit routine
        after confirmation.
        """
        self._cleanup_cache_dir()

    ## Document-specific settings dict.

    @property
    def settings(self):
        """The document-specific settings dict.

        :returns: dict-like object that can be monitored for simple changes.
        :rtype: lib.observable.ObservableDict

        The settings dict is conserved when saving and loading
        OpenRaster files. Note however that a round-trip converts
        strings and dict keys to unicode objects.

        >>> import tempfile, shutil, os.path
        >>> tmpdir = tempfile.mkdtemp()
        >>> doc1 = Document(painting_only=True)
        >>> doc1.settings[u"T.1"] = [1, 2]
        >>> doc1.settings[u"T.2"] = u"simple"
        >>> doc1.settings[u"T.3"] = {u"4": 5}
        >>> expected = [
        ...     (u'T.1', [1, 2]),
        ...     (u'T.2', u'simple'),
        ...     (u'T.3', {u'4': 5}),
        ... ]
        >>> sorted([i for i in doc1.settings.items()
        ...         if i[0].startswith("T.")]) == expected
        True
        >>> file1 = os.path.join(tmpdir, "file1.ora")
        >>> thumb1 = doc1.save(file1)
        >>> doc1.cleanup()
        >>> doc2 = Document(painting_only=True)
        >>> doc2.load(file1)
        True
        >>> sorted([i for i in doc1.settings.items()
        ...         if i[0].startswith("T.")]) == expected
        True
        >>> doc2.settings == doc1.settings
        True
        >>> doc2.cleanup()
        >>> shutil.rmtree(tmpdir)

        See also: ``json``.

        """
        return self._settings

    ## Periodic cache updater

    def _start_cache_updater(self):
        """Start the cache updater if it isn't running."""
        assert not self._painting_only
        assert self._owns_cache_dir
        if self._cache_updater_id:
            return
        logger.debug("cache_updater started")
        self._cache_updater_id = GLib.timeout_add_seconds(
            interval = CACHE_UPDATE_INTERVAL,
            function = self._cache_updater_cb,
        )

    def _stop_cache_updater(self):
        """Stop the cache updater."""
        assert not self._painting_only
        assert self._owns_cache_dir
        if not self._cache_updater_id:
            return
        logger.debug("cache_updater: stopped")
        GLib.source_remove(self._cache_updater_id)
        self._cache_updater_id = None

    def _cache_updater_cb(self):
        """Payload: update canary file, start autosave countdown if dirty"""
        assert not self._painting_only
        assert self._owns_cache_dir
        activity_file_path = os.path.join(self.cache_dir, CACHE_ACTIVITY_FILE)
        os.utime(activity_file_path, None)
        if self._autosave_dirty:
            self._start_autosave_countdown()
        return True

    ## Autosave flag

    @property
    def autosave_backups(self):
        return self._autosave_backups

    @autosave_backups.setter
    def autosave_backups(self, newval):
        newval = bool(newval)
        oldval = bool(self._autosave_backups)
        self._autosave_backups = newval
        if self._painting_only or not self._owns_cache_dir:
            return
        if oldval and not newval:
            self._stop_autosave_writes()
            self._stop_autosave_countdown()

    ## Autosave countdown, restarted by activity.

    def _restart_autosave_countdown(self):
        """Stop and then start the countdown to an automatic backup

        Should be called in response to any user activity which might
        have changed the document's data or its structure.

        """
        assert not self._painting_only
        self._stop_autosave_countdown()
        self._start_autosave_countdown()

    def _start_autosave_countdown(self):
        """Start the countdown to an automatic backup, if it isn't already.

        This does nothing if the countdown has already been started, or
        if the autosave writes are in progress.

        """
        assert not self._painting_only
        if not self._autosave_dirty:
            return
        if self._autosave_processor.has_work():
            return
        if self._autosave_countdown_id:
            return
        if not self._autosave_backups:
            return
        interval = lib.helpers.clamp(self.autosave_interval, 5, 300)
        self._autosave_countdown_id = GLib.timeout_add_seconds(
            interval = interval,
            function = self._autosave_countdown_cb,
        )
        logger.debug(
            "autosave_countdown: autosave will run in %ds",
            self.autosave_interval,
        )

    def _stop_autosave_countdown(self):
        """Stop any existing countdown to an automatic backup"""
        assert not self._painting_only
        if not self._autosave_countdown_id:
            return
        GLib.source_remove(self._autosave_countdown_id)
        self._autosave_countdown_id = None

    def _autosave_countdown_cb(self):
        """Payload: start autosave writes and terminate"""
        assert not self._painting_only
        # Sync settings so they get saved, but not the whole doc.
        # See https://github.com/mypaint/mypaint/issues/893
        self._settings.sync_pending_changes(flush=True)
        self._queue_autosave_writes()
        self._autosave_countdown_id = None
        return False

    ## Queued autosave writes: low priority & chunked

    def _queue_autosave_writes(self):
        """Add autosaved backup tasks to the background processor

        These tasks consist of nicely chunked writes for all layers
        whose data has changed, plus a few extra structural and
        bookkeeping ones.

        """
        if not self._cache_dir:
            logger.warning("autosave start abandoned: _cache_dir not set")
            # sometimes happens on exit
            return
        logger.debug("autosave starting: queueing save tasks")
        assert not self._painting_only
        assert not self._autosave_processor.has_work()
        assert self._autosave_dirty
        oradir = os.path.join(self._cache_dir, CACHE_DOC_AUTOSAVE_SUBDIR)
        datadir = os.path.join(oradir, "data")
        if not os.path.exists(datadir):
            logger.debug("autosave: creating %r...", datadir)
            os.makedirs(datadir)
        # Mimetype entry
        manifest = set()
        with open(os.path.join(oradir, 'mimetype'), 'w') as fp:
            fp.write(lib.xml.OPENRASTER_MEDIA_TYPE)
        manifest.add("mimetype")
        # Dimensions
        image_bbox = tuple(self.get_bbox())
        if self.frame_enabled:
            image_bbox = tuple(self.get_frame())
        # Get root stack element and files that will be needed,
        # queue writes for those files
        taskproc = self._autosave_processor
        root_elem = self.layer_stack.queue_autosave(
            oradir, taskproc, manifest,
            save_srgb_chunks = True,  # internal-only, so sure.
            bbox = image_bbox,
        )
        # Build the image element
        x0, y0, w0, h0 = image_bbox
        image_elem = ET.Element('image')
        image_elem.attrib['w'] = str(w0)
        image_elem.attrib['h'] = str(h0)
        image_elem.attrib[_ORA_EOTF_ATTR] = str(lib.eotf.eotf())
        frame_active_value = ("true" if self.frame_enabled else "false")
        image_elem.attrib[_ORA_FRAME_ACTIVE_ATTR] = frame_active_value
        image_elem.append(root_elem)
        # Store the unsaved painting time too, since recovery needs it.
        # This is a (very) local extension to the format.
        t_str = "{:3f}".format(self.unsaved_painting_time)
        image_elem.attrib[_ORA_UNSAVED_PAINTING_TIME_ATTR] = t_str
        # Doc-specific settings
        settings_file_rel = _ORA_JSON_SETTINGS_ZIP_PATH
        if self._settings is not None:
            taskproc.add_work(
                self._autosave_settings_cb,
                dict(self._settings),
                os.path.join(oradir, settings_file_rel),
            )
            image_elem.attrib[_ORA_JSON_SETTINGS_ATTR] = settings_file_rel
            manifest.add(settings_file_rel)
        # Store version of MyPaint the file was saved with
        image_elem.attrib[_ORA_MYPAINT_VERSION] = lib.meta.MYPAINT_VERSION
        # Thumbnail generation.
        rootstack_sshot = self.layer_stack.save_snapshot()
        rootstack_clone = layer.RootLayerStack(doc=None)
        rootstack_clone.load_snapshot(rootstack_sshot)
        thumbdir_rel = "Thumbnails"
        thumbdir = os.path.join(oradir, thumbdir_rel)
        if not os.path.exists(thumbdir):
            os.makedirs(thumbdir)
        thumbfile_basename = "thumbnail.png"
        thumbfile_rel = os.path.join(thumbdir_rel, thumbfile_basename)
        taskproc.add_work(
            self._autosave_thumbnail_cb,
            rootstack_clone,
            image_bbox,
            os.path.join(thumbdir, thumbfile_basename)
        )
        manifest.add(thumbfile_rel)
        # Final write
        stackfile_rel = "stack.xml"
        taskproc.add_work(
            self._autosave_stackxml_cb,
            image_elem,
            os.path.join(oradir, stackfile_rel),
        )
        manifest.add(stackfile_rel)
        # Cleanup
        taskproc.add_work(
            self._autosave_cleanup_cb,
            oradir = oradir,
            manifest = manifest,
        )

    def _autosave_thumbnail_cb(self, rootstack, bbox, filename):
        """Autosaved backup task: write Thumbnails/thumbnail.png

        This runs every time currently for the same reason we rewrite
        stack.xml each time. It would be a big win if we didn't have to
        do this though.

        """
        assert not self._painting_only
        thumbnail = rootstack.render_thumbnail(bbox)
        tmpname = filename + u".TMP"
        lib.pixbuf.save(thumbnail, tmpname)
        lib.fileutils.replace(tmpname, filename)
        return False

    def _autosave_stackxml_cb(self, image_elem, filename):
        """Autosaved backup task: write stack.xml

        This runs every time because the document's layer structure can
        change without data layers being aware of it.

        """
        assert not self._painting_only
        lib.xml.indent_etree(image_elem)
        tmpname = filename + u".TMP"
        with open(tmpname, 'wb') as xml_fp:
            xml = ET.tostring(image_elem, encoding='UTF-8')
            xml_fp.write(xml)
        lib.fileutils.replace(tmpname, filename)
        return False

    def _autosave_settings_cb(self, settings, filename):
        """Autosaved backup task: save the doc-specific settings dict"""
        assert not self._painting_only

        # Py2/Py3: always feed print() a UTF-8 encoded byte string.
        json_data = json.dumps(settings, indent=2)
        if isinstance(json_data, unicode):
            json_data = json_data.encode("utf-8")
        assert isinstance(json_data, bytes)

        tmpname = filename + u".TMP"
        with open(tmpname, 'wb') as fp:
            fp.write(json_data)
        lib.fileutils.replace(tmpname, filename)

    def _autosave_cleanup_cb(self, oradir, manifest):
        """Autosaved backup task: final cleanup task"""
        assert not self._painting_only
        surplus_files = []
        for dirpath, dirnames, filenames in os.walk(oradir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                filerel = os.path.relpath(filepath, oradir)
                if filerel not in manifest:
                    surplus_files.append(filepath)
        # Remove surplus files.
        # This is fairly normal: it happens when layers are deleted.
        for path in surplus_files:
            logger.debug(
                "autosave: removing %r (not listed in manifest)",
                path,
            )
            os.unlink(path)
        # Also check for files listed in the manifest that aren't
        # present on the disk. That's more of a concern,
        # because it means the index will be inconsistent
        for path in [os.path.join(oradir, p) for p in manifest]:
            if os.path.exists(path):
                continue
            logger.error(
                "autosave: missing %r (listed in the manifest)",
                path,
            )
        self._autosave_dirty = False
        logger.debug("autosave: all done, doc marked autosave-clean")
        return False

    def _stop_autosave_writes(self):
        assert not self._painting_only
        logger.debug("autosave stopped: clearing task queue")
        self._autosave_processor.stop()

    def _command_stack_updated_cb(self, cmdstack):
        assert not self._painting_only
        if not self.autosave_backups:
            return
        self._autosave_dirty = True
        self._restart_autosave_countdown()
        logger.debug("autosave: updates detected, doc marked autosave-dirty")

    ## Document frame

    def get_resolution(self):
        """Returns the document model's nominal resolution

        The OpenRaster format saves resolution information in both vertical and
        horizontal resolutions, but MyPaint does not support this at present.
        This method returns the a unidirectional document resolution in pixels
        per inch; this is the user-chosen factor that UI controls should use
        when converting real-world measurements in frames, fonts, and other
        objects to document pixels.

        Note that the document resolution has no direct relation to screen
        pixels or printed dots.
        """
        if self._xres and self._yres:
            return max(1, max(self._xres, self._yres))
        else:
            return DEFAULT_RESOLUTION

    def set_resolution(self, res):
        """Sets the document model's nominal resolution

        The OpenRaster format saves resolution information in both vertical and
        horizontal resolutions, but MyPaint does not support this at present.
        This method sets the document resolution in pixels per inch in both
        directions.

        Note that the document resolution has no direct relation to screen
        pixels or printed dots.
        """
        if res is not None:
            res = int(res)
            res = max(1, res)
        self._xres = res
        self._yres = res

    def get_frame(self):
        return self._frame

    def set_frame(self, frame, user_initiated=False):
        x, y, w, h = frame
        self.update_frame(x=x, y=y, width=w, height=h,
                          user_initiated=user_initiated)

    frame = property(get_frame, set_frame)

    def update_frame(self, x=None, y=None, width=None, height=None,
                     user_initiated=False):
        """Update parts of the frame"""
        frame = [x, y, width, height]
        if user_initiated:
            if isinstance(self.get_last_command(), command.UpdateFrame):
                self.update_last_command(frame=frame)
            else:
                self.do(command.UpdateFrame(self, frame))
        else:
            new_frame = list(self._frame[:])
            for i, var in enumerate([x, y, width, height]):
                if var is not None:
                    new_frame[i] = int(var)
            if new_frame != self._frame:
                old_frame = tuple(self._frame)
                self._frame[:] = new_frame
                new_frame = tuple(new_frame)
                self.frame_updated(old_frame, new_frame)
                self.effective_bbox_changed()

    @event
    def frame_updated(self, old_frame, new_frame):
        """Event: the frame's dimensions were updated

        :param tuple frame: the new frame extents (x, y, w, h)
        """

    def get_frame_enabled(self):
        return self._frame_enabled

    def set_frame_enabled(self, enabled, user_initiated=False):
        enabled = bool(enabled)
        if self._frame_enabled == enabled:
            return
        if user_initiated:
            self.do(command.SetFrameEnabled(self, enabled))
        else:
            self._frame_enabled = enabled
            self.frame_enabled_changed(enabled)
            self.effective_bbox_changed()

    frame_enabled = property(get_frame_enabled)

    @event
    def frame_enabled_changed(self, enabled):
        """Event: the frame_enabled field changed value"""

    def set_frame_to_current_layer(self, user_initiated=False):
        current = self.layer_stack.current
        x, y, w, h = current.get_bbox()
        self.update_frame(x, y, w, h, user_initiated=user_initiated)

    def set_frame_to_document(self, user_initiated=False):
        x, y, w, h = self.get_bbox()
        self.update_frame(x, y, w, h, user_initiated=user_initiated)

    def trim_current_layer(self):
        """Trim the current layer to the extent of the document frame

        This has no effect if the frame is not currently enabled.

        """
        if not self._frame_enabled:
            return
        self.do(command.TrimLayer(self))

    def uniq_current_layer(self, pixels=False):
        """Udoably remove non-unique tiles or pixels from the current layer."""
        self.do(command.UniqLayer(self, pixels=pixels))

    def refactor_current_layer_group(self, pixels=False):
        """Undoably factor out common parts of child layers to a new child."""
        self.do(command.RefactorGroup(self, pixels=pixels))

    @event
    def effective_bbox_changed(self):
        """Event: the effective bounding box was changed"""

    def _effective_bbox_changed_cb(self, *_ignored):
        # Background layer's autosaved data files depend on the
        # frame's position and size. No other layers need this.
        assert not self._painting_only
        self.layer_stack.background_layer.autosave_dirty = True

    ## Misc actions

    def clear(self, new_cache=True):
        """Clears everything, and resets the command stack

        :param bool new_cache: False to *not* create a new cache dir

        This results in a document consisting of
        one newly created blank drawing layer,
        an empty undo history, and unless `new_cache` is False,
        a new empty working-document temp directory.
        Clearing the document also generates a full redraw,
        and resets the frame, the stored resolution,
        and the document-specific settings.
        """
        self.sync_pending_changes()
        self.layer_view_manager.clear()
        self._layers.symmetry_unset = True
        self._layers.set_symmetry_state(
            False, (0, 0), lib.mypaintlib.SymmetryVertical, 2, 0)
        prev_area = self.get_full_redraw_bbox()
        if self._owns_cache_dir:
            if self._cache_dir is not None:
                self._cleanup_cache_dir()
            if new_cache:
                self._create_cache_dir()
        self.command_stack.clear()
        self._layers.clear()
        if self.CREATE_PAINTING_LAYER_IF_EMPTY:
            self.add_layer((-1,))
            self._layers.current_path = (0,)
            self.command_stack.clear()
        else:
            self._layers.current_path = None
        self.unsaved_painting_time = 0.0
        self.set_frame([0, 0, 0, 0])
        self.set_frame_enabled(False)
        self._xres = None
        self._yres = None
        self._settings.clear()
        self.canvas_area_modified(*prev_area)

    def brushsettings_changed_cb(self, settings):
        self.sync_pending_changes(flush=False)

    def select_layer(self, index=None, path=None, layer=None):
        """Selects a layer undoably"""
        layers = self.layer_stack
        sel_path = layers.canonpath(index=index, path=path, layer=layer,
                                    usecurrent=False, usefirst=True)
        self.do(command.SelectLayer(self, path=sel_path))

    ## Layer stack (z-order and grouping)

    def restack_layer(self, src_path, targ_path):
        """Moves a layer within the layer stack by path, undoably

        :param tuple src_path: path of the layer to be moved
        :param tuple targ_path: target insert path

        The source path must identify an existing layer. The target
        path must be a valid insertion path at the time this method is
        called.
        """
        logger.debug("Restack layer at %r to %r", src_path, targ_path)
        cmd = command.RestackLayer(self, src_path, targ_path)
        self.do(cmd)

    def bubble_current_layer_up(self):
        """Moves the current layer up in the stack (undoable)"""
        cmd = command.BubbleLayerUp(self)
        self.do(cmd)

    def bubble_current_layer_down(self):
        """Moves the current layer down in the stack (undoable)"""
        cmd = command.BubbleLayerDown(self)
        self.do(cmd)

    ## Misc layer command frontends

    def duplicate_current_layer(self):
        """Makes an exact copy of the current layer (undoable)"""
        self.do(command.DuplicateLayer(self))

    def clear_current_layer(self):
        """Clears the current layer (undoable)"""
        rootstack = self.layer_stack
        can_clear = (rootstack.current is not rootstack
                     and not rootstack.current.is_empty())
        if not can_clear:
            return
        self.do(command.ClearLayer(self))

    ## Drawing/painting strokes

    def redo_last_stroke_with_different_brush(self, brushinfo):
        cmd = self.get_last_command()
        if not isinstance(cmd, command.Brushwork):
            return
        cmd.update(brushinfo=brushinfo)

    ## Other painting/drawing

    def flood_fill(
            self, fill_args,
            view_bbox=None, sample_merged=False, src_path=None,
            make_new_layer=False, status_cb=None
    ):
        """Flood-fills a point on the current layer with a color

        :param fill_args: fill arguments object
        :type fill_args: lib.floodfill.FloodFillArguments
        :param view_bbox: Bounding box of the view, restricts fill if present
        :type view_bbox: lib.helpers.Rect
        :param sample_merged: Use all visible layers when sampling
        :type sample_merged: bool
        :param src_path: Path to layer used as reference (if not active layer)
        :type src_path: tuple or None
        :param make_new_layer: Write output to a new layer on top
        :type make_new_layer: bool
        :param status_cb: Gui status/cancellation setup callback

        Filling an infinite canvas requires limits. If the frame is
        enabled, this limits the maximum size of the fill, and filling
        outside the frame is not possible.

        Otherwise, if the entire document is empty, the limits are
        dynamic.  Initially only a single tile will be filled. This can
        then form one corner for the next fill's limiting rectangle.
        This is a little quirky, but allows big areas to be filled
        rapidly as needed on blank layers.
        """
        bbox = helpers.Rect(*tuple(self.get_effective_bbox()))
        if not self.layer_stack.current.get_fillable():
            make_new_layer = True
        if bbox.empty():
            xs = [i[0] for i in fill_args.seeds]
            ys = [i[1] for i in fill_args.seeds]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

            bbox = helpers.Rect()
            bbox.x = N*int(min_x//N)
            bbox.y = N*int(min_y//N)
            bbox.w = N*int(max_x//N) - bbox.x + N
            bbox.h = N*int(max_y//N) - bbox.y + N
        elif not self.frame_enabled:
            for (x, y) in fill_args.seeds:
                bbox.expand_to_include_point(x, y)
        if view_bbox:
            view_bbox = helpers.Rect(*view_bbox)
            if bbox.contains(view_bbox):
                bbox = view_bbox
            elif bbox.overlaps(view_bbox):
                bbox = bbox.intersection(view_bbox)
        fill_args.bbox = bbox
        cmd = command.FloodFill(
            self, fill_args, sample_merged, src_path,
            make_new_layer, status_cb)
        self.do(cmd)

    ## Graphical refresh

    def _canvas_modified_cb(self, root, layer, x, y, w, h):
        """Internal callback: forwards redraw nofifications"""
        self.canvas_area_modified(x, y, w, h)

    @event
    def canvas_area_modified(self, x, y, w, h):
        """Event: canvas was updated, either within a rectangle or fully

        :param x: top-left x coordinate for the redraw bounding box
        :param y: top-left y coordinate for the redraw bounding box
        :param w: width of the redraw bounding box, or 0 for full redraw
        :param h: height of the redraw bounding box, or 0 for full redraw

        This event method is invoked to notify observers about needed redraws
        originating from within the model, e.g. painting, fills, or layer
        moves. It is also used to notify about the entire canvas needing to be
        redrawn. In the latter case, the `w` or `h` args forwarded to
        registered observers is zero.

        See also: `invalidate_all()`.
        """
        pass

    def invalidate_all(self):
        """Marks everything as invalid"""
        self.canvas_area_modified(0, 0, 0, 0)

    ## Undo/redo command stack

    @event
    def sync_pending_changes(self, flush=True, **kwargs):
        """Ask for pending changes to be synchronized (updated/flushed)

        This event is called to signal sources of pending changes that
        they need to synchronize their changes with the document and its
        command stack. Synchronizing normally means that registered
        observers with pending changes:

        * may optionally update their pending changes if needed,
        * must flush their pending changes to the observed model's
          command stack.

        By default, the request to flush changes is non-optional.

        :param bool flush: if this is False, the flush is optional too
        :param \*\*kwargs: passed through to observers

        See: `lib.observable.event` for details of the signalling
        mechanism.

        """

    def _settings_sync_pending_changes_cb(self, doc, flush=True, **kwargs):
        """Make sure the settings get synced when the doc is synced.

        In addition to this, there are times when the doc settings need
        syncing by themselves, e.g. autosave.

        """
        self._settings.sync_pending_changes(flush=flush, **kwargs)

    def undo(self):
        """Undo the most recently done command"""
        self.sync_pending_changes()
        while True:
            cmd = self.command_stack.undo()
            if not cmd or not cmd.automatic_undo:
                return cmd

    def redo(self):
        """Redo the most recently undone command"""
        self.sync_pending_changes()
        while True:
            cmd = self.command_stack.redo()
            if not cmd or not cmd.automatic_undo:
                return cmd

    def do(self, cmd):
        """Do a command"""
        self.sync_pending_changes()
        self.command_stack.do(cmd)

    def update_last_command(self, **kwargs):
        """Updates the most recently done command"""
        self.sync_pending_changes()
        return self.command_stack.update_last_command(**kwargs)

    def get_last_command(self):
        """Gets the the most recently done command"""
        self.sync_pending_changes()
        return self.command_stack.get_last_command()

    ## Utility methods

    def get_bbox(self):
        """Returns the data bounding box of the document

        This is currently the union of all the data bounding boxes of all of
        the layers. It disregards the user-chosen frame.

        """
        res = helpers.Rect()
        for l in self.layer_stack.deepiter():
            # OPTIMIZE: only visible layers?
            bbox = l.get_bbox()
            res.expand_to_include_rect(bbox)
        return res

    def get_full_redraw_bbox(self):
        """Returns the full-redraw bounding box of the document

        This is the same concept as `layer.BaseLayer.get_full_redraw_bbox()`,
        and is built up from the full-redraw bounding boxes of all layers.
        """
        res = helpers.Rect()
        for l in self.layer_stack.deepiter():
            bbox = l.get_full_redraw_bbox()
            if bbox.w == 0 and bbox.h == 0:  # infinite
                res = bbox
            else:
                res.expand_to_include_rect(bbox)
        return res

    def get_effective_bbox(self):
        """Return the effective bounding box of the document.

        If the frame is enabled, this is the bounding box of the frame,
        else the (dynamic) bounding box of the document.

        """
        return self.get_frame() if self.frame_enabled else self.get_bbox()

    def get_user_bbox(self):
        """Return the bounding box expected by the user.

        If the frame is enabled, this is the bounding box of the frame.

        If the frame is disabled, this is a rectangle multiple of the size
        of the current background image, big enough to cover the dynamic
        bounding box of the document. Width and height will always be
        greater than zero, that is, the minimum size is the size of the
        background image.
        """
        if self.frame_enabled:
            return self.get_frame()

        bbox = self.get_bbox()
        bg_bbox = self.layer_stack.background_layer.get_bbox()
        assert bg_bbox.w > 0
        assert bg_bbox.h > 0

        # Note: In Python, -11 % 10 == 9

        correction_x = bbox.x % bg_bbox.w
        if correction_x > 0:
            bbox.x -= correction_x
            bbox.w += correction_x

        correction_y = bbox.y % bg_bbox.h
        if correction_y > 0:
            bbox.y -= correction_y
            bbox.h += correction_y

        if bbox.w == 0:
            bbox.w = bg_bbox.w
        else:
            orphan_w = bbox.w % bg_bbox.w
            if orphan_w > 0:
                bbox.w += bg_bbox.w - orphan_w

        if bbox.h == 0:
            bbox.h = bg_bbox.h
        else:
            orphan_h = bbox.h % bg_bbox.h
            if orphan_h > 0:
                bbox.h += bg_bbox.h - orphan_h

        assert bbox.w > 0
        assert bbox.h > 0
        assert bbox.x % bg_bbox.w == 0
        assert bbox.y % bg_bbox.h == 0
        assert bbox.w % bg_bbox.w == 0
        assert bbox.h % bg_bbox.h == 0

        return bbox

    ## More layer stack commands

    def add_layer(self, path, layer_class=layer.PaintingLayer, **kwds):
        """Undoably adds a new layer at a specified path

        :param path: Path for the new layer
        :param callable layer_class: constructor for the new layer
        :param **kwds: Constructor args

        By default, a normal painting layer is added.

        See: `lib.command.AddLayer`
        """
        cmd = command.AddLayer(
            self, path,
            name=None,
            layer_class=layer_class,
            **kwds
        )
        self.do(cmd)

    def remove_current_layer(self):
        """Delete the current layer"""
        if not self.layer_stack.current_path:
            return
        self.do(command.RemoveLayer(self))

    def rename_current_layer(self, name):
        """Rename the current layer"""
        if not self.layer_stack.current_path:
            return
        cmd_class = command.RenameLayer
        cmd = self.get_last_command()
        layer = self.layer_stack.current
        if isinstance(cmd, cmd_class) and cmd.layer is layer:
            logger.info("Updating the last layer rename: %r", name)
            self.update_last_command(name=name)
        else:
            cmd = cmd_class(self, name, layer=layer)
            self.do(cmd)

    def normalize_layer_mode(self):
        """Normalize current layer's mode and opacity"""
        layers = self.layer_stack
        self.do(command.NormalizeLayerMode(self, layers.current))

    def merge_current_layer_down(self):
        """Merge the current layer into the one below"""
        rootstack = self.layer_stack
        cur_path = rootstack.current_path
        if cur_path is None:
            return False
        dst_path = rootstack.get_merge_down_target(cur_path)
        if dst_path is None:
            logger.info("Merge Down is not possible here")
            return False
        self.do(command.MergeLayerDown(self))
        return True

    def merge_visible_layers(self):
        """Merge all visible layers into one & discard originals."""
        self.do(command.MergeVisibleLayers(self))

    def new_layer_merged_from_visible(self):
        """Combine all visible layers into a new one & keep originals"""
        self.do(command.NewLayerMergedFromVisible(self))

    ## Layer import/export

    def load_layer_from_pixbuf(self, pixbuf, x=0, y=0, to_new_layer=False):
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        s = tiledsurface.Surface()
        bbox = s.load_from_numpy(arr, x, y)
        self.do(command.LoadLayer(self, s, to_new_layer=to_new_layer))
        return bbox

    def load_layer_from_png(self, filename, x, y, progress=None,
                            **kwargs):
        s = tiledsurface.Surface()
        bbox = s.load_from_png(filename, x, y, progress, **kwargs)
        self.do(command.LoadLayer(self, s))
        return bbox

    def update_layer_from_external_edit_tempfile(self, layer, file_path):
        """Update a layer after external edits to its tempfile"""
        assert hasattr(layer, "load_from_external_edit_tempfile")
        cmd = command.ExternalLayerEdit(self, layer, file_path)
        self.do(cmd)

    ## Even more layer command frontends

    def set_layer_visibility(self, visible, layer):
        """Sets the visibility of a layer."""
        if layer is self.layer_stack:
            return
        cmd_class = command.SetLayerVisibility
        cmd = self.get_last_command()
        if isinstance(cmd, cmd_class) and cmd.layer is layer:
            self.update_last_command(visible=visible)
        else:
            cmd = cmd_class(self, visible, layer)
            self.do(cmd)

    def set_layer_locked(self, locked, layer):
        """Sets the input-locked status of a layer."""
        if layer is self.layer_stack:
            return
        cmd_class = command.SetLayerLocked
        cmd = self.get_last_command()
        if isinstance(cmd, cmd_class) and cmd.layer is layer:
            self.update_last_command(locked=locked)
        else:
            cmd = cmd_class(self, locked, layer)
            self.do(cmd)

    def set_current_layer_opacity(self, opacity):
        """Sets the opacity of the current layer

        :param float opacity: New layer opacity
        """
        current = self.layer_stack.current
        if current is self.layer_stack:
            return
        if current.mode == layer.PASS_THROUGH_MODE:
            return
        cmd_class = command.SetLayerOpacity
        cmd = self.get_last_command()
        if isinstance(cmd, cmd_class) and cmd.layer is current:
            logger.debug("Updating current layer opacity: %r", opacity)
            self.update_last_command(opacity=opacity)
        else:
            logger.debug("Setting current layer opacity: %r", opacity)
            cmd = cmd_class(self, opacity, layer=current)
            self.do(cmd)

    def set_current_layer_mode(self, mode):
        """Sets the mode for the current layer

        :param int mode: New layer mode to use
        """
        current = self.layer_stack.current
        if current is self.layer_stack:
            return
        logger.debug("Setting current layer mode: %r", mode)
        cmd = command.SetLayerMode(self, mode, layer=current)
        self.do(cmd)

    ## Saving and loading

    def load_from_pixbuf(self, pixbuf, to_new_layer=False):
        """Load a document from a pixbuf."""
        self.clear()
        bbox = self.load_layer_from_pixbuf(pixbuf, to_new_layer=to_new_layer)
        self.set_frame(bbox, user_initiated=False)

    def save(self, filename, **kwargs):
        """Save the document to a file.

        :param str filename: The filename to save to.
        :param dict kwargs: Passed on to the chosen save method.
        :raise lib.error.FileHandlingError: with a good user-facing string
        :raise lib.error.AllocationError: with a good user-facing string
        :returns: A thumbnail pixbuf, or None if not supported
        :rtype: GdkPixbuf

        The filename's extension is used to determine the save format, and a
        ``save_*()`` method is chosen to perform the save.
        """
        self.sync_pending_changes(flush=True)
        junk, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        save = getattr(self, 'save_' + ext, self._unsupported)
        result = None
        try:
            result = save(filename, **kwargs)
        except GObject.GError as e:
            logger.exception("GError when writing %r: %s", filename, e)
            if e.code == 5:
                # add a hint due to a very consfusing error message when
                # there is no space left on device
                hint_tmpl = C_(
                    "Document IO: hint templates for user-facing exceptions",
                    u'Unable to write “{filename}”: {err}\n'
                    u'Do you have enough space left on the device?'
                )
            else:
                hint_tmpl = C_(
                    "Document IO: hint templates for user-facing exceptions",
                    u'Unable to write “{filename}”: {err}'
                )
            raise FileHandlingError(hint_tmpl.format(
                filename = filename,
                err = e,
            ))
        except IOError as e:
            logger.exception("IOError when writing %r: %s", filename, e)
            hint_tmpl = C_(
                "Document IO: hint templates for user-facing exceptions",
                u'Unable to write “{filename}”: {err}'
            )
            raise FileHandlingError(hint_tmpl.format(
                filename = filename,
                err = e,
            ))
        self.unsaved_painting_time = 0.0
        return result

    def load(self, filename, **kwargs):
        """Load the document from a file.

        :param str filename:
            The filename to load from. The extension is used to determine
            format, and a ``load_*()`` method is chosen to perform the load.
        :param dict kwargs:
            Passed on to the chosen loader method.
        :raise FileHandlingError: with a suitable string

        >>> doc = Document()
        >>> doc.load("tests/smallimage.ora")
        True
        >>> doc.cleanup()

        """
        error_kwargs = {
            "error_loading_common": _LOAD_FAILED_COMMON_TEMPLATE_LINE.format(
                basename = os.path.basename(filename),
                filename = filename,
            ),
            "see_logs": _ERROR_SEE_LOGS_LINE,
            "filename": filename,
            "basename": os.path.basename(filename),
        }
        if not os.path.isfile(filename):
            msg = C_(
                "Document IO: loading errors",
                u"{error_loading_common}\n"
                u"The file does not exist."
            ).format(**error_kwargs)
            raise FileHandlingError(msg)
        if not os.access(filename, os.R_OK):
            msg = C_(
                "Document IO: loading errors",
                u"{error_loading_common}\n"
                u"You do not have the permissions needed "
                u"to open this file."
            ).format(**error_kwargs)
            raise FileHandlingError(msg)
        junk, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        load_method_name = 'load_' + ext
        load_method = getattr(self, load_method_name, self._unsupported)
        logger.debug(
            "Using %r to load %r (kwargs=%r)",
            load_method_name,
            filename,
            kwargs,
        )
        error_str = None
        result = None
        try:
            result = load_method(filename, **kwargs)
        except (GObject.GError, IOError) as e:
            logger.exception("Error when loading %r", filename)
            error_str = unicode(e)
        except Exception as e:
            logger.exception("Failed to load %r", filename)
            tmpl = C_(
                "Document IO: loading errors",
                u"{error_loading_common}\n\n"
                u"Reason: {reason}\n\n"
                u"{see_logs}"
            )
            error_kwargs["reason"] = unicode(e)
            error_str = tmpl.format(**error_kwargs)
        if error_str:
            raise FileHandlingError(error_str)
        self.command_stack.clear()
        self.unsaved_painting_time = 0.0
        return result

    def _unsupported(self, filename, *args, **kwargs):
        stemname, ext = os.path.splitext(filename)
        error_kwargs = {
            "error_loading_common": _LOAD_FAILED_COMMON_TEMPLATE_LINE.format(
                basename = os.path.basename(filename),
                filename = filename,
            ),
            "ext": ext,
            "basename": os.path.basename(filename),
            "filename": filename,
        }
        tmpl = C_(
            "Document IO: loading errors",
            u"{error_loading_common}\n"
            u"Unknown file format extension: “{ext}”"
        )
        raise FileHandlingError(tmpl.format(**error_kwargs))

    def import_layers(self, filenames, progress=None, **kwargs):
        """Imports layers at the current position from files.

        >>> doc = Document()
        >>> len(doc.layer_stack)
        1
        >>> doc.import_layers([
        ...    "tests/smallimage.ora",
        ...    "tests/bigimage.ora",
        ... ])
        >>> len(doc.layer_stack)
        2
        >>> doc.cleanup()

        """
        if progress is None:
            progress = lib.feedback.Progress()
        progress.items = len(filenames)

        logger.info(
            "Importing layers from %d file(s) via a temporary document",
            len(filenames),
        )
        import_group = layer.LayerStack()
        import_group.name = C_(
            "Document IO: group name for Import Layers",
            u"Imported layers",
        )
        try:
            tmp_doc = Document(cache_dir=self._cache_dir)
            for filename in filenames:
                tmp_doc.load(
                    filename,
                    progress=progress.open(),
                    **kwargs
                )
                tmp_root = tmp_doc.layer_stack

                layers = list(tmp_root)
                if len(layers) == 0:
                    return
                elif len(layers) == 1:
                    # Single-layer .ora and .png go directly into
                    # the import group.
                    targ_group = import_group
                    if not layers[0].has_interesting_name():
                        layers[0].name = os.path.basename(filename)
                else:
                    # A multi-layer .ora files gets a subgroup of its own,
                    # named after the imported file.
                    targ_group = layer.LayerStack()
                    targ_group.name = os.path.basename(filename)
                    import_group.append(targ_group)

                for child_layer in layers:
                    tmp_root.remove(child_layer)
                    targ_group.append(child_layer)

                tmp_doc.clear()
        finally:
            tmp_doc.cleanup()

        path = self.layer_stack.current_path
        cmd = command.AddLayer(self, path, layer=import_group, is_import=True)
        self.do(cmd)
        progress.close()

    def render_thumbnail(self, **kwargs):
        """Renders a thumbnail for the user bbox"""
        t0 = time.time()
        bbox = self.get_user_bbox()
        pixbuf = self.layer_stack.render_thumbnail(bbox, **kwargs)
        logger.info('Rendered thumbnail in %d seconds.',
                    time.time() - t0)
        return pixbuf

    def save_png(self, filename, alpha=None, multifile=None, progress=None,
                 **kwargs):
        """Save to one or more PNG files"""
        if progress is None:
            progress = lib.feedback.Progress()

        if multifile == "layers":
            if alpha is None:
                alpha = True
            self._save_layers_to_numbered_pngs(filename, alpha, progress,
                                               **kwargs)
        elif multifile == "views":
            if alpha is None:
                alpha = not self.layer_stack.background_visible
            self._save_layer_views_to_named_pngs(filename, alpha, progress,
                                                 **kwargs)
        elif multifile is not None:
            raise ValueError("only valid multifile values: 'layers', 'views'")
        else:
            if alpha is None:
                alpha = not self.layer_stack.background_visible
            self._save_single_file_png(filename, alpha, progress, **kwargs)

    def _save_single_file_png(self, filename, alpha, progress, **kwargs):
        """Save to a single PNG, with optional alpha."""
        x, y, w, h = self.get_user_bbox()

        self.layer_stack.save_as_png(
            filename,
            x, y, w, h,
            alpha=alpha,
            render_background=not alpha,
            progress=progress,
            **kwargs
        )

    def _save_layers_to_numbered_pngs(self, filename, alpha, progress,
                                      **kwargs):
        """Save layers to multiple number-suffixed PNG files."""
        prefix, ext = os.path.splitext(filename)

        # if we have a number already, strip it
        s = prefix.rsplit('.', 1)
        if s[-1].isdigit():
            prefix = s[0]

        x, y, w, h = self.get_user_bbox()

        layers = [lr for path, lr in self.layer_stack.walk()]
        progress.items = len(layers)
        for i, lr in enumerate(layers):
            filename = '%s.%03d%s' % (prefix, i+1, ext)
            lr.save_as_png(
                filename,
                x, y, w, h,
                alpha=alpha,
                progress=progress.open(),
                **kwargs
            )

    def _save_layer_views_to_named_pngs(self, filename, alpha, progress,
                                        **kwargs):
        """Save the layer-views to multiple name-suffixed PNG files."""
        prefix, ext = os.path.splitext(filename)

        lvm = self.layer_view_manager
        old_active_view = lvm.current_view_name
        all_views = sorted(lvm.view_names)
        view_was_changed = False

        try:
            progress.items = len(all_views)
            used_namefrags = set()
            for view_name in all_views:
                frag = safename(view_name, fragment=True)
                frag = make_unique_name(frag, used_namefrags)
                used_namefrags.add(frag)
                filename = "{prefix}.{view_name}{ext}".format(
                    prefix=prefix,
                    view_name=frag,
                    ext=ext,
                )
                lvm.activate_view_by_name(view_name)
                view_was_changed = True
                self._save_single_file_png(
                    filename, alpha,
                    progress=progress.open(),
                    **kwargs
                )

        finally:
            if view_was_changed:
                lvm.activate_view_by_name(old_active_view)

    def load_png(self, filename, progress=None, **kwargs):
        """Load (speedily) from a PNG file"""
        self.clear()
        bbox = self.load_layer_from_png(filename, 0, 0, progress, **kwargs)
        self.set_frame(bbox, user_initiated=False)

    def load_from_pixbuf_file(self, filename, progress=None, **kwargs):
        """Load from a file which GdkPixbuf can open"""
        pixbuf = lib.pixbuf.load_from_file(filename, progress)
        self.load_from_pixbuf(pixbuf)

    load_jpg = load_from_pixbuf_file
    load_jpeg = load_from_pixbuf_file

    @fileutils.via_tempfile
    def save_jpg(self, filename, quality=90, **kwargs):
        bbox = self.get_user_bbox()
        root = self.layer_stack
        try:
            pixbuf = root.render_layer_as_pixbuf(root, bbox, **kwargs)
        except (AllocationError, MemoryError) as e:
            hint_tmpl = C_(
                "Document IO: hint templates for user-facing exceptions",
                u"Unable to save as JPEG: {original_msg}\n\n"
                u"Try saving in PNG format instead, "
                u"if your machine doesn’t have a lot of memory. "
                u"MyPaint’s PNG save function is more efficient."
            )
            raise AllocationError(hint_tmpl.format(
                original_msg = str(e),
            ))
        lib.pixbuf.save(pixbuf, filename, 'jpeg', quality=str(quality))

    save_jpeg = save_jpg

    @fileutils.via_tempfile
    def save_ora(self, filename, options=None, **kwargs):
        """Saves OpenRaster data to a file"""
        logger.info('save_ora: %r (%r, %r)', filename, options, kwargs)
        t0 = time.time()
        self.sync_pending_changes(flush=True)
        thumbnail = _save_layers_to_new_orazip(
            self.layer_stack,
            filename,
            bbox=tuple(self.get_user_bbox()),
            xres=self._xres if self._xres else None,
            yres=self._yres if self._yres else None,
            frame_active = self.frame_enabled,
            settings=dict(self._settings),
            **kwargs
        )
        logger.info('%.3fs save_ora total', time.time() - t0)
        return thumbnail

    @staticmethod
    def _compat_check(image_elem, filename, **kwargs):
        target_version = image_elem.attrib.get(_ORA_MYPAINT_VERSION, None)
        if not target_version:
            return True
        result = lib.meta.compatibility(target_version)
        if not result:
            return True
        compat_type, prerel = result

        def ignore(*a, **kw):
            return True
        cb = kwargs.get('incompatible_ora_cb', ignore)
        return cb(compat_type, prerel, filename, target_version)

    def load_ora(self, filename, progress=None, **kwargs):
        """Loads from an OpenRaster file"""
        logger.info('load_ora: %r', filename)
        t0 = time.time()
        self.clear()
        cache_dir = self._cache_dir
        orazip = zipfile.ZipFile(filename)
        logger.debug('mimetype: %r', orazip.read('mimetype').strip())
        xml = orazip.read('stack.xml')
        image_elem = ET.fromstring(xml)
        root_stack_elem = image_elem.find('stack')
        # Compatibility check
        if not Document._compat_check(image_elem, filename, **kwargs):
            return False

        image_width = max(0, int(image_elem.attrib.get('w', 0)))
        image_height = max(0, int(image_elem.attrib.get('h', 0)))
        # Resolution: false value, 0 specifically, means unspecified
        image_xres = max(0, int(image_elem.attrib.get('xres', 0)))
        image_yres = max(0, int(image_elem.attrib.get('yres', 0)))

        # Determine which compatibility mode the file should be opened with
        eotf = image_elem.attrib.get(_ORA_EOTF_ATTR, None)
        if 'compat_handler' in kwargs:
            kwargs['compat_handler'](eotf, root_stack_elem)

        # Delegate loading of image data to the layers tree itself
        self.layer_stack.load_from_openraster(
            orazip,
            root_stack_elem,
            cache_dir,
            progress,
            x=0, y=0,
            invert_strokemaps=(eotf is None),
            **kwargs
        )
        assert len(self.layer_stack) > 0

        # Resolution information if specified
        # Before frame to benefit from its observer call
        if image_xres and image_yres:
            self._xres = image_xres
            self._yres = image_yres
        else:
            self._xres = None
            self._yres = None

        # Set the frame size to that saved in the image.
        self.update_frame(x=0, y=0, width=image_width, height=image_height,
                          user_initiated=False)

        # Enable frame if the image was saved with its frame active.
        frame_enab = lib.xml.xsd2bool(
            image_elem.attrib.get(_ORA_FRAME_ACTIVE_ATTR, "false"),
        )
        self.set_frame_enabled(frame_enab, user_initiated=False)

        # Document-specific settings dict.
        self._settings.clear()
        json_entry = image_elem.attrib.get(_ORA_JSON_SETTINGS_ATTR, None)
        if json_entry is not None:
            new_settings = {}
            try:
                # Py3: on our Travis-CI, they're using Ubuntu Trusty's
                # ancient Python 3.4.0, and that has a regression. Need
                # to always feed that version unicode strings.
                # Normally json.loads() doesn't care, provided that any
                # bytes it sees are UTF-8. Which they always have been.
                json_str = orazip.read(json_entry)
                if isinstance(json_str, bytes):
                    json_str = json_str.decode("utf-8")
                new_settings = json.loads(json_str)
            except Exception:
                logger.exception(
                    "Failed to load JSON settings from zipfile's %r entry",
                    json_entry,
                )
            self._settings.update(new_settings)

        orazip.close()

        logger.info('%.3fs load_ora total', time.time() - t0)
        return True

    def resume_from_autosave(self, autosave_dir, progress=None):
        """Resume using an autosave dir (and its parent cache dir)"""
        assert os.path.isdir(autosave_dir)
        assert os.path.basename(autosave_dir) == CACHE_DOC_AUTOSAVE_SUBDIR
        doc_cache_dir = os.path.dirname(autosave_dir)
        app_cache_dir = get_app_cache_root()
        assert (os.path.basename(os.path.dirname(doc_cache_dir)) ==
                os.path.basename(app_cache_dir))
        self._stop_cache_updater()
        self._stop_autosave_writes()
        self.clear(new_cache=False)
        try:
            self._load_from_openraster_dir(
                autosave_dir,
                doc_cache_dir,
                progress=progress,
                retain_autosave_info=True,
            )
        except Exception as e:
            # Assign a valid *new* cache dir before bailing out.
            assert self._cache_dir is None
            self.clear(new_cache=True)
            # Log, and tell the user about it
            logger.exception("Failed to resume from %r", autosave_dir)
            tmpl = C_(
                "Document autosave: restoring: errors",
                u"Failed to recover work from an automated backup.\n\n"
                u"Reason: {reason}\n\n"
                u"{see_logs}"
            )
            raise FileHandlingError(
                tmpl.format(
                    app_cache_root = app_cache_dir,
                    reason = unicode(e),
                    see_logs = _ERROR_SEE_LOGS_LINE,
                ),
                investigate_dir = doc_cache_dir,
            )
        else:
            self._cache_dir = doc_cache_dir

    def _load_from_openraster_dir(self, oradir, cache_dir,
                                  progress=None,
                                  retain_autosave_info=False,
                                  **kwargs):
        """Load from an OpenRaster-style folder.

        :param unicode oradir: Directory with a .ORA-like structure
        :param unicode cache_dir: Doc cache for storing layer revs etc.
        :param progress: Unsized progress object: updates UI.
        :type progress: lib.feedback.Progress or None
        :param bool retain_autosave_info: Restore unsaved time etc.
        :param \*\*kwargs: Passed through to layer loader methods.

        The oradir folder is treated as read-only during this operation.

        """
        self.clear()
        with open(os.path.join(oradir, "mimetype"), "r") as fp:
            logger.debug('mimetype: %r', fp.read().strip())
        doc = ET.parse(os.path.join(oradir, "stack.xml"))
        image_elem = doc.getroot()
        width = max(0, int(image_elem.attrib.get('w', 0)))
        height = max(0, int(image_elem.attrib.get('h', 0)))
        xres = max(0, int(image_elem.attrib.get('xres', 0)))
        yres = max(0, int(image_elem.attrib.get('yres', 0)))
        # Delegate layer loading to the layers tree.
        root_stack_elem = image_elem.find("stack")
        self.layer_stack.load_from_openraster_dir(
            oradir,
            root_stack_elem,
            cache_dir,
            progress,
            x=0, y=0,
            **kwargs
        )
        assert len(self.layer_stack) > 0
        if retain_autosave_info:
            self.unsaved_painting_time = max(0.0, float(
                image_elem.attrib.get(_ORA_UNSAVED_PAINTING_TIME_ATTR, 0.0)
            ))
        # Resolution information if specified
        # Before frame to benefit from its observer call
        if xres and yres:
            self._xres = xres
            self._yres = yres
        else:
            self._xres = None
            self._yres = None
        # Set the frame size to that saved in the image.
        self.update_frame(x=0, y=0, width=width, height=height,
                          user_initiated=False)
        # Enable frame if the image was saved with its frame active.
        frame_enab = lib.xml.xsd2bool(
            image_elem.attrib.get(_ORA_FRAME_ACTIVE_ATTR, "false"),
        )
        self.set_frame_enabled(frame_enab, user_initiated=False)

        # Document-specific settings dict.
        self._settings.clear()
        json_rel = image_elem.attrib.get(_ORA_JSON_SETTINGS_ATTR, None)
        if json_rel is not None:
            json_path = os.path.join(oradir, json_rel)
            new_settings = {}
            try:
                with open(json_path, 'rb') as fp:
                    json_data = fp.read()
                    json_data = json_data.decode("utf-8")
                    # Py3: see note in load_ora().
                    new_settings = json.loads(json_data)
            except Exception:
                logger.exception(
                    "Failed to load JSON settings from %r",
                    json_path,
                )
            self._settings.update(new_settings)

    ## Layer visibility sets

    @property
    def layer_view_manager(self):
        """RO property: the layer visibility set manager for this doc."""
        return self._layer_view_manager


def _save_layers_to_new_orazip(root_stack, filename, bbox=None,
                               xres=None, yres=None,
                               frame_active=False,
                               progress=None,
                               settings=None,
                               **kwargs):
    """Save a root layer stack to a new OpenRaster zipfile

    :param lib.layer.RootLayerStack root_stack: what to save
    :param unicode filename: where to save
    :param tuple bbox: area to save, None to use the inherent data bbox
    :param int xres: nominal X resolution for the doc
    :param int yres: nominal Y resolution for the doc
    :param frame_active: True if the frame is enabled
    :param progress: Unsized UI feedback object
    :type progress: lib.feedback.Progress or None
    :param \*\*kwargs: Passed through to root_stack.save_to_openraster()
    :rtype: GdkPixbuf
    :returns: Thumbnail preview image (256x256 max) of what was saved

    >>> from lib.gibindings import GdkPixbuf
    >>> from lib.layer.test import make_test_stack
    >>> root, leaves = make_test_stack()
    >>> import tempfile
    >>> tmpdir = tempfile.mkdtemp()
    >>> orafile = os.path.join(tmpdir, "test.ora")
    >>> thumb = _save_layers_to_new_orazip(root, orafile, settings={
    ...     "thing.one": 42,
    ...     "thing.two": [101, 99],
    ... })
    >>> isinstance(thumb, GdkPixbuf.Pixbuf)
    True
    >>> assert os.path.isfile(orafile)
    >>> shutil.rmtree(tmpdir)
    >>> assert not os.path.exists(tmpdir)

    """

    if not progress:
        progress = lib.feedback.Progress()
    progress.items = 100

    tempdir = tempfile.mkdtemp(suffix='mypaint', prefix='save')
    if not isinstance(tempdir, unicode):
        tempdir = tempdir.decode(sys.getfilesystemencoding())

    orazip = zipfile.ZipFile(
        filename, 'w',
        compression=zipfile.ZIP_STORED,
    )

    # The mimetype entry must be first
    helpers.zipfile_writestr(orazip, 'mimetype', lib.xml.OPENRASTER_MEDIA_TYPE)

    # Update the initially-selected flag on all layers
    # Also get the data bounding box as we go
    data_bbox = helpers.Rect()
    for s_path, s_layer in root_stack.walk():
        selected = (s_path == root_stack.current_path)
        s_layer.initially_selected = selected
        data_bbox.expand_to_include_rect(s_layer.get_bbox())
    data_bbox = tuple(data_bbox)

    # First 90%: save the layer stack
    image = ET.Element('image')
    if bbox is None:
        bbox = data_bbox
    x0, y0, w0, h0 = bbox
    image.attrib['w'] = str(w0)
    image.attrib['h'] = str(h0)
    image.attrib[_ORA_EOTF_ATTR] = str(lib.eotf.eotf())
    root_stack_path = ()
    root_stack_elem = root_stack.save_to_openraster(
        orazip, tempdir, root_stack_path,
        data_bbox, bbox,
        progress=progress.open(90),
        **kwargs
    )
    image.append(root_stack_elem)

    # Frame-enabled state
    frame_active_value = ("true" if frame_active else "false")
    image.attrib[_ORA_FRAME_ACTIVE_ATTR] = frame_active_value

    # Document-specific settings dict.
    if settings is not None:

        # Py2/Py3: always feed writestr() a UTF-8 encoded byte string.
        json_data = json.dumps(dict(settings), indent=2)
        if isinstance(json_data, unicode):
            json_data = json_data.encode("utf-8")
        assert isinstance(json_data, bytes)

        zip_path = _ORA_JSON_SETTINGS_ZIP_PATH
        helpers.zipfile_writestr(orazip, zip_path, json_data)
        image.attrib[_ORA_JSON_SETTINGS_ATTR] = zip_path

    # MyPaint version
    image.attrib[_ORA_MYPAINT_VERSION] = lib.meta.MYPAINT_VERSION

    # Resolution info
    if xres and yres:
        image.attrib["xres"] = str(xres)
        image.attrib["yres"] = str(yres)

    # OpenRaster version declaration
    image.attrib["version"] = lib.xml.OPENRASTER_VERSION

    # Last 10%: previews.
    # Thumbnail preview (256x256)
    thumbnail = root_stack.render_thumbnail(
        bbox,
        progress=progress.open(1),
    )
    tmpfile = join(tempdir, 'tmp.png')
    lib.pixbuf.save(thumbnail, tmpfile, 'png')
    orazip.write(tmpfile, 'Thumbnails/thumbnail.png')
    os.remove(tmpfile)

    # Save fully rendered image too
    tmpfile = os.path.join(tempdir, "mergedimage.png")
    root_stack.save_as_png(
        tmpfile, *bbox,
        alpha=False, background=True,
        progress=progress.open(9),
        **kwargs
    )
    orazip.write(tmpfile, 'mergedimage.png')
    os.remove(tmpfile)

    # Prettification
    lib.xml.indent_etree(image)
    xml = ET.tostring(image, encoding='UTF-8')

    # Finalize
    helpers.zipfile_writestr(orazip, 'stack.xml', xml)
    orazip.close()
    os.rmdir(tempdir)

    progress.close()
    return thumbnail


def get_app_cache_root():
    """Get the app-specific cache root dir, creating it if needed.

    :returns: The cache folder root for the app.
    :rtype: unicode

    Document-specific cache folders go inside this.

    """
    cache_root = lib.glib.get_user_cache_dir()
    app_cache_root = os.path.join(cache_root, CACHE_APP_SUBDIR_NAME)
    if not os.path.exists(app_cache_root):
        logger.debug("Creating %r", app_cache_root)
        os.makedirs(app_cache_root)
    assert isinstance(app_cache_root, unicode)
    return app_cache_root


def get_available_autosaves():
    """Get all known autosaves

    :returns: a sequence of AutosaveInfo instances
    :rtype: iterable

    For use with autosave recovery dialogs.

    See: Document.resume_from_autosave().

    """
    app_cache_root = get_app_cache_root()
    for doc_cache_name in os.listdir(app_cache_root):
        if not doc_cache_name.startswith(CACHE_DOC_SUBDIR_PREFIX):
            continue
        autosave_path = os.path.join(
            app_cache_root,
            doc_cache_name,
            CACHE_DOC_AUTOSAVE_SUBDIR,
        )
        if not os.path.isdir(autosave_path):
            continue
        yield AutosaveInfo.new_for_path(autosave_path)


def _get_path_mtime(path):
    return datetime.fromtimestamp(os.stat(path).st_mtime)
