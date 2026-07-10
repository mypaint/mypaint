#!/usr/bin/env python3

"""Unit tests for layout-independent keyboard accelerator handling."""

import importlib
import sys
import types
import unittest

_MISSING = object()


class ModifierType(int):
    pass


ModifierType.SHIFT_MASK = ModifierType(1 << 0)
ModifierType.LOCK_MASK = ModifierType(1 << 1)
ModifierType.CONTROL_MASK = ModifierType(1 << 2)
ModifierType.MOD1_MASK = ModifierType(1 << 3)
ModifierType.META_MASK = ModifierType(1 << 4)


class FakeGdk:
    ModifierType = ModifierType

    @staticmethod
    def keyval_to_lower(keyval):
        return ord(chr(keyval).lower())

    @staticmethod
    def unicode_to_keyval(codepoint):
        return codepoint


class FakeGtk:
    @staticmethod
    def accelerator_get_default_mod_mask():
        return ModifierType(
            ModifierType.SHIFT_MASK
            | ModifierType.CONTROL_MASK
            | ModifierType.MOD1_MASK
            | ModifierType.META_MASK
        )


class Event:
    def __init__(self, keycode, state=0, group=0):
        self.hardware_keycode = keycode
        self.state = ModifierType(state)
        self.group = group


class KeymapKey:
    def __init__(self, group):
        self.group = group


class Keymap:
    def __init__(self, translations, option_groups=()):
        self.translations = translations
        self.option_groups = set(option_groups)
        self.calls = []

    def get_entries_for_keycode(self, keycode):
        groups = []
        keyvals = []
        for (entry_keycode, group), keyval in self.translations.items():
            if entry_keycode == keycode and group not in groups:
                groups.append(group)
                keyvals.append(keyval)
        return bool(groups), [KeymapKey(g) for g in groups], keyvals

    def translate_keyboard_state(self, keycode, state, group):
        self.calls.append((keycode, state, group))
        keyval = self.translations.get((keycode, group))
        if keyval is None:
            return None
        consumed = ModifierType.SHIFT_MASK
        if group in self.option_groups:
            consumed |= ModifierType.MOD1_MASK
        if state & ModifierType.SHIFT_MASK:
            keyval = ord(chr(keyval).upper())
        return True, keyval, group, 0, consumed


class KeyboardUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gui

        cls._gui_package = gui
        cls._old_gui_attribute = getattr(gui, "keyboardutils", _MISSING)
        cls._old_bindings = sys.modules.get("lib.gibindings")
        cls._old_module = sys.modules.pop("gui.keyboardutils", None)

        bindings = types.ModuleType("lib.gibindings")
        bindings.Gdk = FakeGdk
        bindings.Gtk = FakeGtk
        sys.modules["lib.gibindings"] = bindings
        cls.module = importlib.import_module("gui.keyboardutils")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("gui.keyboardutils", None)
        if cls._old_module is not None:
            sys.modules["gui.keyboardutils"] = cls._old_module
        if cls._old_gui_attribute is _MISSING:
            delattr(cls._gui_package, "keyboardutils")
        else:
            cls._gui_package.keyboardutils = cls._old_gui_attribute
        if cls._old_bindings is None:
            sys.modules.pop("lib.gibindings", None)
        else:
            sys.modules["lib.gibindings"] = cls._old_bindings

    def setUp(self):
        self.module._IS_DARWIN = True

    def test_macos_uses_event_group_instead_of_option_group(self):
        keymap = Keymap(
            {(0x01, 0): ord("s"), (0x01, 1): ord("ß")},
            option_groups={1},
        )
        result = self.module.get_key_event_accelerators(Event(0x01), keymap)
        self.assertEqual(result, [(ord("s"), ModifierType(0))])
        self.assertEqual(keymap.calls, [(0x01, ModifierType(0), 0)])

    def test_macos_non_latin_layout_gets_physical_latin_fallback(self):
        keymap = Keymap({(0x01, 0): ord("ы")})
        result = self.module.get_key_event_accelerators(Event(0x01), keymap)
        self.assertEqual(result[0], (ord("ы"), ModifierType(0)))
        self.assertEqual(result[1], (ord("s"), ModifierType(0)))

    def test_active_layout_has_priority_over_physical_fallback(self):
        keymap = Keymap({(0x00, 0): ord("q")})
        result = self.module.get_key_event_accelerators(Event(0x00), keymap)
        self.assertEqual(result[0], (ord("q"), ModifierType(0)))
        self.assertEqual(result[1], (ord("a"), ModifierType(0)))

    def test_shifted_letter_keeps_shift_modifier(self):
        keymap = Keymap({(0x01, 0): ord("ы")})
        event = Event(0x01, ModifierType.SHIFT_MASK)
        result = self.module.get_key_event_accelerators(event, keymap)
        self.assertEqual(
            result[1],
            (ord("s"), ModifierType.SHIFT_MASK),
        )

    def test_shifted_symbol_uses_symbol_without_shift_modifier(self):
        keymap = Keymap({(0x18, 0): ord("=")})
        event = Event(0x18, ModifierType.SHIFT_MASK)
        result = self.module.get_key_event_accelerators(event, keymap)
        self.assertEqual(result[-1], (ord("+"), ModifierType(0)))

    def test_option_is_preserved_for_physical_fallback(self):
        keymap = Keymap({(0x01, 1): ord("ß")}, option_groups={1})
        event = Event(0x01, ModifierType.MOD1_MASK, group=1)
        result = self.module.get_key_event_accelerators(event, keymap)
        self.assertEqual(result[0], (ord("ß"), ModifierType(0)))
        self.assertEqual(
            result[1],
            (ord("s"), ModifierType.MOD1_MASK),
        )

    def test_caps_lock_is_ignored(self):
        keymap = Keymap({(0x01, 0): ord("s")})
        event = Event(0x01, ModifierType.LOCK_MASK)
        result = self.module.get_key_event_accelerators(event, keymap)
        self.assertEqual(result, [(ord("s"), ModifierType(0))])
        self.assertEqual(keymap.calls[0][1], ModifierType(0))

    def test_unknown_keycode_has_no_accelerator(self):
        keymap = Keymap({})
        result = self.module.get_key_event_accelerators(Event(0x7F), keymap)
        self.assertEqual(result, [])

    def test_other_gdk_groups_are_layout_fallbacks(self):
        self.module._IS_DARWIN = False
        keymap = Keymap(
            {
                (0x01, 0): ord("s"),
                (0x01, 1): ord("ы"),
                (0x01, 2): ord("ش"),
            }
        )
        result = self.module.get_key_event_accelerators(
            Event(0x01, group=1),
            keymap,
        )
        self.assertEqual(
            result,
            [
                (ord("ы"), ModifierType(0)),
                (ord("s"), ModifierType(0)),
                (ord("ش"), ModifierType(0)),
            ],
        )
        self.assertEqual([call[2] for call in keymap.calls], [1, 0, 2])

    def test_failed_active_group_falls_back_to_another_group(self):
        self.module._IS_DARWIN = False
        keymap = Keymap({(0x01, 1): ord("s")})
        result = self.module.get_key_event_accelerators(
            Event(0x01, group=0),
            keymap,
        )
        self.assertEqual(result, [(ord("s"), ModifierType(0))])
        self.assertEqual([call[2] for call in keymap.calls], [0, 1])

    def test_control_modifier_is_preserved_across_layouts(self):
        self.module._IS_DARWIN = False
        keymap = Keymap({(0x01, 0): ord("s"), (0x01, 1): ord("ы")})
        event = Event(0x01, ModifierType.CONTROL_MASK, group=1)
        result = self.module.get_key_event_accelerators(event, keymap)
        self.assertEqual(
            result,
            [
                (ord("ы"), ModifierType.CONTROL_MASK),
                (ord("s"), ModifierType.CONTROL_MASK),
            ],
        )

    def test_duplicate_accelerators_from_groups_are_removed(self):
        self.module._IS_DARWIN = False
        keymap = Keymap({(0x01, 0): ord("s"), (0x01, 1): ord("s")})
        result = self.module.get_key_event_accelerators(Event(0x01), keymap)
        self.assertEqual(result, [(ord("s"), ModifierType(0))])


if __name__ == "__main__":
    unittest.main()
