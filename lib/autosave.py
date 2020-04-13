# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Autosave/autorecover interface"""

from __future__ import division, print_function

import abc
import uuid

import lib.idletask  # noqa


class Autosaveable:
    """Mixin and abstract base for auto-saveable structures"""

    __metaclass__ = abc.ABCMeta

    @property
    def autosave_dirty(self):
        """Flag indicating that a resave's needed

        :rtype: bool

        Setting this indicates that the in-memory data has changed, and
        data file writes should be queued on the next run-through of
        queue_autosave. Not every layer type needs this.

        The initial value is True.
        """
        try:
            return self.__autosave_dirty
        except AttributeError:
            self.__autosave_dirty = True
            return self.__autosave_dirty

    @autosave_dirty.setter
    def autosave_dirty(self, value):
        """Setter for the dirty flag"""
        self.__autosave_dirty = bool(value)

    @property
    def autosave_uuid(self):
        """UUID prefix for autosave (layer) data filenames

        Naming an object's data files by UUID means that
        queued tasks don't have to know the overall tree structure,
        and also that layer instances
        always update the same set of files over their lifetime.

        """
        try:
            return self.__autosave_uuid
        except AttributeError:
            self.__autosave_uuid = str(uuid.uuid4())
            return self.__autosave_uuid

    @abc.abstractmethod
    def queue_autosave(self, oradir, taskproc, manifest, bbox, **kwargs):
        """Queue tasks for auto-recovery saving

        This kind of save updates a directory structure resembling the
        contents of an OpenRaster zipfile with data from the layer and
        all its child layers. It's an update, so if the data hasn't
        changed since the file was last written, re-writing the file can
        be skipped.

        :param unicode oradir: Root of OpenRaster-like structure
        :param lib.idletask.Processor taskproc: Output: queue of tasks
        :param set manifest: Output: files in data/ to retain afterward
        :param tuple bbox: frame bounding box, (x,y,w,h)
        :param \*\*kwargs: To be passed to underlying save routines.
        :rtype: xml.etree.ElementTree.Element
        :returns: Element indexing files when written (for stack.xml)

        When implementations of this method are invoked on an
        Autosaveable object, the object should:

        1. Update the manifest with details of its files,
        2. Determine if its files need updating,
        3. If its files need updating,
           queue one or more task callbacks which will do that,
        4. Call this method on its children,
        5. Return an XML element for stack.xml
           which indexes its own files,
           and includes the XML elements from its children.

        Auto-recovery saving needs to be split into small tasks so that
        it doesn't interfere too much with foreground processing.
        Individual PNG tile strips or small file copies have about the
        right granularity.

        It follows that snapshots should be used for auto-saving,
        because the user can make changes between the queued tasks
        as the queue is run.

        The returned element should contain sub-elements for any
        sub-layers, and the queue operation should recursively call this
        method on its sub-layers, with the same output queue and
        manifest.

        Paths added to the manifest are relative to oradir. Tasks are
        expected to create these if they don't exist, and the file names
        should use autosave_uuid.
        File writes should be atomic (use lib.file.replace).

        (Yes, this is the visitor pattern made all baroque. Roll on
        Python 3 when we'll be able to do something like
        https://tavianator.com/2014/06/the-visitor-pattern-in-python/)

        """
