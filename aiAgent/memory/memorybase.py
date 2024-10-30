# -*- coding: utf-8 -*-
""" Base class for memory """
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Sequence, Literal, List,Callable,Iterable

from ..msg.message import Msg


# memory 基类
# memory 要提供，add， delete ， get_memory ，clear ，size
class MemoryBase(ABC):

    _version: int = 1

    # 增加
    @abstractmethod
    def add(
        self,
        memories: Union[Sequence[Msg], Msg, None],
    ) -> None:
        """
        Adding new memory fragment, depending on how the memory are stored
        Args:
            memories (Union[Sequence[Msg], Msg, None]):
                Memories to be added.
        """

    # 删除
    @abstractmethod
    def delete(self, index: Union[Iterable, int]) -> None:
        """
        Delete memory fragment, depending on how the memory are stored
        and matched
        Args:
            index (Union[Iterable, int]):
                indices of the memory fragments to delete
        """

    # 查
    @abstractmethod
    def get_memory(
        self,
        recent_n: Optional[int] = None,
        filter_func: Optional[Callable[[int, dict], bool]] = None,
    ) -> list:
        """
        Return a certain range (`recent_n` or all) of memory,
        filtered by `filter_func`
        Args:
            recent_n (int, optional):
                indicate the most recent N memory pieces to be returned.
            filter_func (Optional[Callable[[int, dict], bool]]):
                filter function to decide which pieces of memory should
                be returned, taking the index and a piece of memory as
                input and return True (return this memory) or False
                (does not return)
        """

    # 清空
    @abstractmethod
    def clear(self) -> None:
        """Clean memory, depending on how the memory are stored"""

    # 获取消息总数
    @abstractmethod
    def size(self) -> int:
        """Returns the number of memory segments in memory."""
        raise NotImplementedError