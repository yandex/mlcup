# typing imports
from omegaconf import DictConfig
from typing import Any, Union, Optional

# generic imports
import pydoc
import logging


__all__ = [
    'instantiate',
    'ClassDescription',
]


logger = logging.getLogger(__name__)



class ClassDescription(DictConfig):
    """Just a dummy class for typing of configs containing some class path and class.
    Typical instance of ClassDesription looks like this:
        {
            'cls': 'package.module1.module2.ClsName',
            'args': {'arg1': <smth>, 'arg2': <smth>},
            'some_other_field': <smth>
        }
    """
    pass


def _locate(path: str) -> Any:
    result = pydoc.locate(path)
    if result is None:
        raise ValueError(f"failed to find class: {path}")
    return result


def instantiate(config: Optional[Union[str, ClassDescription]], *args, **extra_kwargs):
    """Instantiates class given by `config.cls` with
    optional args given by `config.args`
    """
    try:
        if config is None:
            return None
        elif type(config) is str:
            cls = _locate(config)
            return cls()
        else:
            cls = _locate(config['cls'])
            kwargs = config.get('args', dict())

            return cls(
                *args,
                **extra_kwargs,
                **kwargs
            )
    except:
        logger.exception(f"Could not instantiate from config {config}")
        raise
