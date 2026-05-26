import pydoc
import sys
from importlib import import_module
from pathlib import Path
from typing import Union
from yacs.config import CfgNode as CN
from addict import Dict


class ConfigDict(Dict):
    """
    Dict wrapper used internally by py2dict.
    """

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return value
        raise ex


def py2dict(file_path: Union[str, Path]) -> dict:
    """Dynamically import a Python config file and return its module-level variables."""
    file_path = Path(file_path).absolute()

    if file_path.suffix != ".py":
        raise TypeError(f"Only Py file can be parsed, but got {file_path.name} instead.")

    if not file_path.exists():
        raise FileExistsError(f"There is no file at the path {file_path}")

    module_name = file_path.stem

    if "." in module_name:
        raise ValueError("Dots are not allowed in config file path.")

    config_dir = str(file_path.parent)

    sys.path.insert(0, config_dir)

    mod = import_module(module_name)
    sys.path.pop(0)

    # Clean up sys.modules to ensure the module is re-executed on next load.
    if module_name in sys.modules:
        del sys.modules[module_name]

    cfg_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}

    return cfg_dict


def py2cfg(file_path: Union[str, Path]) -> CN:
    """
    Load a YACS CfgNode from a Python config file by calling its get_config() function.
    """
    cfg_dict = py2dict(file_path)

    if 'get_config' not in cfg_dict or not callable(cfg_dict['get_config']):
        raise ValueError(
            f"Config file at {file_path} must define and expose a callable 'get_config' function."
        )

    get_config_func = cfg_dict['get_config']

    class MockArgs:
        def __init__(self):
            self.opts = []
            self.batch_size = None

    yacs_config = get_config_func(MockArgs())

    if not isinstance(yacs_config, CN):
        raise TypeError(
            f"The 'get_config' function must return a yacs.config.CfgNode, but it returned {type(yacs_config)}."
        )

    yacs_config.defrost()

    return yacs_config


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)

    return pydoc.locate(object_type)(**kwargs)
