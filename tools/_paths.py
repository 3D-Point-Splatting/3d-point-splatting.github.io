#!/usr/bin/env python3
"""Where the generators read their source material from.

This page is published; the trees it is GENERATED FROM are not. Their
locations therefore come from the environment instead of being baked into
the source, which would publish a private directory layout and the
filenames of unreleased material:

    TDPS_ROOT   the 3DPS research tree  (assets/, output/teaser_panels/)
    DECK_ROOT   the slide-deck tree     (blender/, slides/figsrc/, slides/figs/)

Set whichever the tool you are running needs:

    export TDPS_ROOT=/path/to/mm3DGS
    export DECK_ROOT=/path/to/a_exam

Paths INSIDE this repository resolve relative to this file (REPO), so they
need no environment variable.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def root(name):
    """An external source tree, from the environment.

    Deliberately has no default: a wrong one would not fail, it would
    quietly generate the page from the wrong data.
    """
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"{name} is not set.\n"
            f"The generators read from trees that are not part of this "
            f"repository; see tools/_paths.py.\n"
            f"  export {name}=/path/to/tree")
    v = os.path.expanduser(v)
    if not os.path.isdir(v):
        raise SystemExit(f"{name}={v} is not a directory")
    return v
