# -*- coding: utf-8 -*-
""" Base class for memory """
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Sequence, Literal, List,Callable,Iterable, cast
from pathlib import Path
import aiofiles
from aiofiles.ospath import exists
from aiofiles.os import remove

from ..logger import LOG_INFO,LOG_ERROR,LOG_WARNING,LOG_DEBUG,LOG_CRITICAL
from .memorybase import MemoryBase
from ..msg.message import Msg, MessageBase

# memory 内存类的存储 临时缓存
# 
class TemporaryMemory(MemoryBase):
    #文件路径
    _root_dir: str

    #文件编码方式
    _encoding: str

    def __init__(self):
        """Init method definition."""
 
        # 设置本地存储,消息都是顺序存储的，获取的时候也是指定序号获取的,所以使用 集合比较方便
        self._content = []

   # 向memory中添加数据，用户消息,输入的是 Msg
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
        if memories is None:
            return

        if not isinstance(memories, Sequence):
            record_memories = [memories]
        else:
            record_memories = memories

        # if memory doesn't have id attribute, we skip the checking
        # 创建一个集合 (set)，其中包含了 self._content集合中 所有具有 id 属性的对象的 id 值 
        # 我们的代码没有占位符处理，不存在冲突的情况
        # memories_idx = set(_.id for _ in self._content if hasattr(_, "id"))

        #遍历输入的消息内容
        for memory_unit in record_memories:
            # 如果输入内容不是 message 结构，转成message结构
            if not issubclass(type(memory_unit), MessageBase):
                try:
                    memory_unit = Msg(**memory_unit) #把消息内容展开成 msg格式
                except Exception as exc:
                    raise ValueError(
                        f"Cannot add {memory_unit} to memory, "
                        f"must be with subclass of MessageBase",
                    ) from exc

            # add to memory if it's new
            #if (
            #    not hasattr(memory_unit, "id")
            #    or memory_unit.id not in memories_idx
            #):
            # message结构将内容之间存储到 _content 中去
            self._content.append(memory_unit)

    # 删除
    def delete(self, index: Union[Iterable, int]) -> None:

        if self.size() == 0:
            return

        if isinstance(index, int):
            index = [index]

        if isinstance(index, list):
            index = set(index)

            #索引不能小于0大于等于大小
            invalid_index = [_ for _ in index if _ >= self.size() or _ < 0]
            if len(invalid_index) > 0:
                LOG_INFO("delete : invalid_index",invalid_index)
            
            #除了在 index中其他的都存起来， index中的就删除掉
            self._content = [
                _ for i, _ in enumerate(self._content) if i not in index
            ]
        else:
            raise NotImplementedError(
                "index type only supports {None, int, list}",
            )


    # 查
    def get_memory(
        self,
        recent_n: Optional[int] = None,
        filter_func: Optional[Callable[[int, dict], bool]] = None,
    ) -> list:
        # extract the recent `recent_n` entries in memories
        if recent_n is None:# 如果 recent_n 参数为None，返回所有的消息
            memories = self._content
        else:
            if recent_n > self.size():
                LOG_INFO("get_memory : recent_n",recent_n)
                LOG_INFO("get_memory : size()",self.size())
            memories = self._content[-recent_n:]

        # filter the memories
        if filter_func is not None:
            memories = [_ for i, _ in enumerate(memories) if filter_func(i, _)]

        return memories

    # 清空
    def clear(self) -> None:
        """Clean memory, depending on how the memory are stored"""
        self._content = []

    # 获取消息总数
    def size(self) -> int:
        """Returns the number of memory segments in memory."""
        return len(self._content) # 消息总数
    