"""
Agent orchestration module — 系统集成组维护

包含:
  - types.py:    所有模块共用的枚举与数据结构
  - state.py:    LangGraph PipelineAgentState (TypedDict)
  - interfaces.py: 三大模块的接口契约 (Protocol)
  - demo.py:     Pipeline 概念验证 Demo

各组对接流程:
  1. 阅读 interfaces.py 中本组的 Protocol 契约
  2. 按契约实现模块 (函数签名必须一致)
  3. 集成组将模块包装为 LangGraph node 插入 pipeline
"""

from .types import (
    ActionType,
    DetectedObject,
    ExecutionResult,
    LocationSpec,
    LocationType,
    ObjectClass,
    ParsedInstruction,
    SceneObservation,
    StepResult,
    TaskPlan,
    TaskStep,
    TaskStatus,
)
from .state import PipelineAgentState
from .interfaces import BIN_POSITIONS, PRIMITIVES

__all__ = [
    "PipelineAgentState",
    # Constants
    "BIN_POSITIONS",
    "PRIMITIVES",
    # Enums
    "ActionType",
    "ObjectClass",
    "TaskStatus",
    "LocationType",
    # Data structures
    "ParsedInstruction",
    "LocationSpec",
    "DetectedObject",
    "SceneObservation",
    "TaskStep",
    "TaskPlan",
    "StepResult",
    "ExecutionResult",
]
