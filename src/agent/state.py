"""
PipelineAgentState — LangGraph 状态定义

AgentState 是在 LangGraph 节点间流动的唯一数据载体。
每个节点 (NLP/Vision/Planning/Execute) 读取自己需要的字段，
返回自己负责更新的字段。

设计原则:
  1. 顶层字段用 TypedDict 定义, LangGraph 原生支持
  2. 每个字段由唯一模块负责写入, 避免多写冲突
  3. 字段名与 types.py 中数据类的字段名一致, 转换时零心智负担
"""

import operator
from typing import Annotated, TypedDict


class PipelineAgentState(TypedDict, total=False):
    # ── 用户输入 ───────────────────────────────────────
    instruction: str                # (外部输入) 自然语言指令

    # ── NLP 模块写入 ───────────────────────────────────
    parsed_intent: str              # ActionType.value  e.g. "pick", "move"
    target_object: str              # 目标物体类别  e.g. "screwdriver"
    target_attributes: dict         # 属性约束  e.g. {"color": "red"}
    source_location: dict | None    # LocationSpec.to_dict()
    target_location: dict | None    # LocationSpec.to_dict()
    constraints: list[str]          # 额外约束
    nlp_confidence: float           # NLP 解析置信度

    # ── 视觉模块写入 ──────────────────────────────────
    scene_id: str                   # 场景标识
    detected_objects: list[dict]    # [DetectedObject.to_dict()]
    frame_size: tuple[int, int]     # (w, h) — 图像分辨率, 视觉模块写入

    # ── 规划模块写入 ──────────────────────────────────
    task_id: str                    # 任务唯一 ID
    task_steps: list[dict]          # [TaskStep.to_dict()]
    estimated_duration: float       # 预估耗时

    # ── 执行模块写入 ──────────────────────────────────
    step_results: list[dict]        # [StepResult.to_dict()]
    overall_status: str             # TaskStatus.value
    error_message: str | None       # 错误信息

    # ── 执行日志 (Annotated + add = 追加而非覆盖) ─────
    logs: Annotated[list[str], operator.add]
