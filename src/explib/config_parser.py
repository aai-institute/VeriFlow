from copy import deepcopy
from importlib import import_module
from pathlib import Path
from pickle import load
from typing import Any, Dict, List, Union

import torch
import yaml
# Convenience import for direct access in config files via "__eval__"
from ray import tune


def unfold_raw_config(d: Dict[str, Any]):
    """Unfolds an ordered DAG given as a dictionary into a tree given as dictionary.
    That means that unfold_dict(d) is bisimilar to d but no two distinct key paths in the resulting
    dictionary reference the same object

    :param d: The dictionary to unfold
    """
    du = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            du[k] = unfold_raw_config(v)
        elif isinstance(v, list):
            du[k] = [unfold_raw_config(x) for x in v]
        else:
            du[k] = deepcopy(v)

    return du


def push_overwrites(item: Any, attributes: Dict[str, Any]) -> Any:
    """Pushes the overwrites in the given dictionary to the given item.

    If the item already specifies an overwrite, it is updated.
    If the item is a dictionary, an overwrite specification for the dictionary is created.
    If the item is a list, the overwrites are pushed each element and the processed list is returned.
    Otherwise, item is overwritten by attributen

    :param item: The item to push the overwrites to.
    :param overwrites: The overwrites to push.
    """
    try:
        if "__exact__" in attributes:
            return deepcopy(attributes["__exact__"])
    except:
        pass

    if isinstance(item, dict):
        if "__overwrites__" not in item:
            result = deepcopy(attributes)
            result["__overwrites__"] = item
        else:
            result = item
            result["__overwrites__"].update(attributes)
    elif isinstance(item, list):
        result = [push_overwrites(x, attributes) for x in item]
    else:
        result = deepcopy(attributes)

    return result


def apply_overwrite(d: Dict[str, Any], recurse: bool = True):
    """Applies the "__overwrites__" keyword sematic to a unfolded raw config dictionary and returns the result.
    Except for the special semantics that applies to dictionaries and lists (see below),
    all keys $k$ that are present in in the  "__overwrites__" dictionary $o$ are overwritten in $d$ by $o[k]$.

    ** Dict/List overwrites **:
    - If $d[k]$ is a dictionary, then $d[k]$ must be a dictionary and overwrites of $o[k]$ are recursively
    to the corresponding $d[k]$, i.e. lower-lever overwrites are specified/updated (see notes on recursion).
    The only exception is if $o[k]$ contains the special key "__exact__" with value True. In this case
    $d[k$]$ is replaced by $o[k]["__exact__"]$.
    - If $o[k]$ is a list, then $o[k]$ is pushed to all list elements.

    ** Recursion **:
    If recursion is enabled, overwrites are are fully expanded in infix order where nested overwrites
    (see behavior on dict/list overwrites) are pushed (and overwrite) to the next level, i.e.
    higher level overwrites lower level. Else, only the top-level overwrites are applied.

    ** Note **: Applying this function to a non-unfolded dictionary
    can result in unexpected behavior due to side side-effects.

    :param d: The unfolded raw config dictionary.
    :pram recurse: If True, the overwrites are applied recursively. Defaults to True.
            Can be useful for efficient combination of this method with other parsing methods.
    """

    # Apply top-level overwrite
    if "__overwrites__" in d:
        overwritten_attr = d
        d = overwritten_attr.pop("__overwrites__")

        for k, v in overwritten_attr.items():
            if k not in d:
                d[k] = v
            else:
                d[k] = push_overwrites(d[k], v)

    if recurse:
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = apply_overwrite(v)
            elif isinstance(v, list):
                d[k] = [apply_overwrite(x) for x in v]

    return d


def read_config(yaml_path: Union[str, Path]) -> dict:
    """Loads a yaml file and returns the corresponding dictionary.
    Besides the standard yaml syntax, the function also supports the following
    additional functionality:

    Special keys:
    __class__: The value is interpreted as a class name and the corresponding class is imported.
    __object__: The value is interpreted as a class, all other keys are interpreted as constructor arguments.
        The key indicates that this (sub-)dictionary is interpreted as on object specification.
    __eval__: The value is evaluated. All other keys in the (sub-)dictionary are ignored.
        The keywords supports the core python languages. Additionally, tune and torch are already imported for convenience.
    Example:
        ---
        entry in yaml:
        model:
            __class__: src.verfiflow.flows.NiceFlow
        entry in result: model: <src.verfiflow.flows.NiceFlow>
        ---
        entry in yaml:
        model:
            __object__: src.verfiflow.flows.NiceFlow
            p1: 1
            p2: 2
        entry in result: model: <src.verfiflow.flows.NiceFlow(p1=1, p2=2)>
        ---
        entry in yaml:
        lr:
            __eval__: tune.loguniform(1e-4, 1e-1)
        entry in result: lr: <tune.loguniform(1e-4, 1e-1)>

    :param yaml_path: Path to the yaml file.
    """

    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = unfold_raw_config(config)
    config = parse_raw_config(config)

    return config


def parse_raw_config(d: dict) -> Any:
    """Parses an unfolded raw config dictionary and returns the corresponding dictionary.
    Parsing includes the following steps:
    - Overwrites are applied (see apply_overwrite)
    - The "__object__" key is interpreted as a class name and the corresponding class is imported.
    - The "__eval__" key is evaluated.
    - The "__class__" key is interpreted as a class name and the corresponding class is imported.

    Args:
        d: The raw config dictionary.
    Returns:
        The result after all semantics have been applied.
    """
    if isinstance(d, dict):
        d = apply_overwrite(d, recurse=False)

        # Depth-first recursion
        for k, v in d.items():
            d[k] = parse_raw_config(v)

        if "__object__" in d:
            module, cls = d["__object__"].rsplit(".", 1)
            C = getattr(import_module(module), cls)
            d.pop("__object__")
            return C(**d)
        elif "__eval__" in d:
            return eval(d["__eval__"])
        elif "__class__" in d:
            module, cls = d["__class__"].rsplit(".", 1)
            C = getattr(import_module(module), cls)
            return C
        else:
            return d
    elif isinstance(d, list):
        result = [parse_raw_config(x) for x in d]
        return result
    else:
        return d


def from_checkpoint(params: str, state_dict: str) -> Any:
    """Loads a model from a checkpoint.

    Args:
        params: Path to the file containing the model specification.
        state_dict: Path to the file containing the state dict.
    Returns:
        The loaded model.
    """
    spec = load(open(params, "rb"))["model_cfg"]
    model = spec["type"](**spec["params"])

    state_dict = torch.load(state_dict)
    model.load_state_dict(state_dict)

    return model
