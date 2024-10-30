# -*- coding: utf-8 -*-
""" 导入包中所有pipeline相关模块"""
from .pipeline import (
    PipelineBase,
    SequentialPipeline,
    IfElsePipeline,
    SwitchPipeline,
    ForLoopPipeline,
    WhileLoopPipeline,
)

from .functional import (
    sequentialpipeline,
    ifelsepipeline,
    switchpipeline,
    forlooppipeline,
    whilelooppipeline,
)

__all__ = [
    "PipelineBase",
    "SequentialPipeline",
    "IfElsePipeline",
    "SwitchPipeline",
    "ForLoopPipeline",
    "WhileLoopPipeline",
    "sequentialpipeline",
    "ifelsepipeline",
    "switchpipeline",
    "forlooppipeline",
    "whilelooppipeline",
]
