from __future__ import annotations
from collections.abc import Iterator
from typing import Callable
from xarray.core.treenode import Tree
'These iterators are copied from anytree.iterators, with minor modifications.'

class LevelOrderIter(Iterator):
    """Iterate over tree applying level-order strategy starting at `node`.
    This is the iterator used by `DataTree` to traverse nodes.

    Parameters
    ----------
    node : Tree
        Node in a tree to begin iteration at.
    filter_ : Callable, optional
        Function called with every `node` as argument, `node` is returned if `True`.
        Default is to iterate through all ``node`` objects in the tree.
    stop : Callable, optional
        Function that will cause iteration to stop if ``stop`` returns ``True``
        for ``node``.
    maxlevel : int, optional
        Maximum level to descend in the node hierarchy.

    Examples
    --------
    >>> from xarray.core.datatree import DataTree
    >>> from xarray.core.iterators import LevelOrderIter
    >>> f = DataTree(name="f")
    >>> b = DataTree(name="b", parent=f)
    >>> a = DataTree(name="a", parent=b)
    >>> d = DataTree(name="d", parent=b)
    >>> c = DataTree(name="c", parent=d)
    >>> e = DataTree(name="e", parent=d)
    >>> g = DataTree(name="g", parent=f)
    >>> i = DataTree(name="i", parent=g)
    >>> h = DataTree(name="h", parent=i)
    >>> print(f)
    <xarray.DataTree 'f'>
    Group: /
    ├── Group: /b
    │   ├── Group: /b/a
    │   └── Group: /b/d
    │       ├── Group: /b/d/c
    │       └── Group: /b/d/e
    └── Group: /g
        └── Group: /g/i
            └── Group: /g/i/h
    >>> [node.name for node in LevelOrderIter(f)]
    ['f', 'b', 'g', 'a', 'd', 'i', 'c', 'e', 'h']
    >>> [node.name for node in LevelOrderIter(f, maxlevel=3)]
    ['f', 'b', 'g', 'a', 'd', 'i']
    >>> [
    ...     node.name
    ...     for node in LevelOrderIter(f, filter_=lambda n: n.name not in ("e", "g"))
    ... ]
    ['f', 'b', 'a', 'd', 'i', 'c', 'h']
    >>> [node.name for node in LevelOrderIter(f, stop=lambda n: n.name == "d")]
    ['f', 'b', 'g', 'a', 'i', 'h']
    """

    def __init__(self, node: Tree, filter_: Callable | None=None, stop: Callable | None=None, maxlevel: int | None=None):
        self.node = node
        self.filter_ = filter_
        self.stop = stop
        self.maxlevel = maxlevel
        self.__iter = None

    def __iter__(self) -> Iterator[Tree]:
        return self

    def __next__(self) -> Iterator[Tree]:
        if self.__iter is None:
            self.__iter = self.__init()
        item = next(self.__iter)
        return item