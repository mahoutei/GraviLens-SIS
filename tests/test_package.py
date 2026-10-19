import importlib
import pkgutil

import gravilens_sis


def test_every_module_imports():
    mods = [m.name for m in pkgutil.walk_packages(gravilens_sis.__path__,
                                                  gravilens_sis.__name__ + '.')]
    assert mods, 'no submodules discovered'
    for m in mods:
        importlib.import_module(m)
