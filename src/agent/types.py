"""
接口类型定义 — 系统集成组维护

本文件定义了工业智能体 pipeline 中所有模块间的数据契约。
模块按此契约独立开发，集成时只需接入对应输入输出。

坐标系约定:
  - World frame: 桌面中心为原点, X→前, Y→左, Z→上 (单位: 米)
  - Pixel frame: 图像左上角为原点, X→右, Y→下 (单位: 像素)
  - 料箱编号: bin_1 ~ bin_6, 世界坐标由仿真组统一配置

更新记录:
  - 2026-06-27: 初版, 定义 PipelineAgentState 及三大模块接口契约
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ═══════════════════════════════════════════════════════════════════
# 枚举 — 统一的动作/物体/状态词汇表
# ═══════════════════════════════════════════════════════════════════


class ActionType(str, Enum):
    """指令意图类型 — NLP 模块输出"""
    PICK = "pick"               # 抓取
    PLACE = "place"             # 放置
    MOVE = "move"               # 搬运 (pick + place 组合)
    SORT = "sort"               # 分拣
    COUNT = "count"             # 计数
    INSPECT = "inspect"         # 检测/检查
    UNKNOWN = "unknown"


class ObjectClass(str, Enum):
    """工业工具/零件类别 — 视觉 & NLP 共用"""
    SCREWDRIVER = "screwdriver"
    WRENCH = "wrench"
    NUT = "nut"
    BEARING = "bearing"
    ROLLER = "roller"
    PLIER = "plier"
    HAMMER = "hammer"
    BOLT = "bolt"
    UNKNOWN = "unknown"


class TaskStatus(str, Enum):
    """任务/步骤执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class LocationType(str, Enum):
    """位置标识类型"""
    BIN = "bin"         # 料箱编号, e.g. "bin_3"
    TABLE = "table"     # 桌面
    COORD_3D = "coord"  # 世界坐标系位姿


# ═══════════════════════════════════════════════════════════════════
# 数据结构 — 各模块的输入/输出契约
# ═══════════════════════════════════════════════════════════════════


# ── NLP 模块输出 ──────────────────────────────────────────────────

@dataclass
class LocationSpec:
    """位置描述 — 自然语言→结构化位置"""
    type: LocationType
    label: str = ""            # e.g. "bin_3", "table_left"
    coord: list[float] | None = None  # [x, y, z] world coords (米)

    def to_dict(self) -> dict:
        return {"type": self.type.value, "label": self.label, "coord": self.coord}


@dataclass
class ParsedInstruction:
    """NLP 模块输出: 结构化指令

    NLP 组的核心交付物。输入原始中文指令, 输出此结构体。
    """
    raw_text: str                              # 原始指令文本
    intent: ActionType = ActionType.UNKNOWN    # 动作意图
    target_object: str = ""                    # 目标物体类别名 (ObjectClass 值)
    target_attributes: dict = field(default_factory=dict)  # 属性约束 {"color": "red", "size": "large"}
    source_location: LocationSpec | None = None  # 源位置 (place时可为None)
    target_location: LocationSpec | None = None  # 目标位置 (pick时可为None)
    constraints: list[str] = field(default_factory=list)   # 额外约束 ["左侧", "第三个"]
    confidence: float = 0.0                    # 解析置信度 [0, 1]

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "intent": self.intent.value,
            "target_object": self.target_object,
            "target_attributes": self.target_attributes,
            "source_location": self.source_location.to_dict() if self.source_location else None,
            "target_location": self.target_location.to_dict() if self.target_location else None,
            "constraints": self.constraints,
            "confidence": self.confidence,
        }


# ── 视觉模块输出 ──────────────────────────────────────────────────

@dataclass
class DetectedObject:
    """单个检测结果 — 视觉模块对场景中一个物体的描述

    视觉组的核心交付物。每个物体包含 2D 图像坐标 + 3D 世界坐标。
    """
    id: str                    # 唯一标识
    class_name: str            # 类别名 (ObjectClass 值)
    bbox_2d: list[float]       # 像素框 [x1, y1, x2, y2]
    position_3d: list[float]   # 世界坐标 [x, y, z] 米
    confidence: float          # 置信度 [0, 1]
    orientation: list[float] | None = None   # [roll, pitch, yaw] 弧度, 可选
    dimensions: list[float] | None = None    # [w, h, d] 米, 可选

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "bbox_2d": self.bbox_2d,
            "position_3d": self.position_3d,
            "confidence": self.confidence,
            "orientation": self.orientation,
            "dimensions": self.dimensions,
        }


@dataclass
class SceneObservation:
    """视觉模块输出: 完整场景描述

    视觉组每次推理的完整输出。集成组用 scene_id 做缓存/去重。
    """
    scene_id: str                              # 场景标识 (时间戳或帧号)
    objects: list[DetectedObject] = field(default_factory=list)
    timestamp: float = 0.0                     # 采集/推理时刻
    frame_size: tuple[int, int] = (640, 480)   # 图像分辨率

    def get_by_class(self, class_name: str) -> list[DetectedObject]:
        """按类别筛选物体"""
        return [o for o in self.objects if o.class_name == class_name]

    def get_best_match(self, class_name: str) -> DetectedObject | None:
        """返回置信度最高的匹配"""
        matches = self.get_by_class(class_name)
        return max(matches, key=lambda o: o.confidence) if matches else None

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "objects": [o.to_dict() for o in self.objects],
            "timestamp": self.timestamp,
            "frame_size": list(self.frame_size),
        }


# ── 规划/执行模块输出 ─────────────────────────────────────────────

@dataclass
class TaskStep:
    """单个原子动作 — 规划组的核心交付物

    每一步必须包含: step_id (递增), primitive (原语动作名), params (参数)。
    """
    step_id: int               # 步骤序号 (从 1 开始)
    primitive: str             # 原子原语: move_to, open_gripper, close_gripper, ...
    params: dict = field(default_factory=dict)  # 原语参数
    description: str = ""      # 人类可读描述
    timeout: float = 5.0       # 超时时间 (秒)
    retry_on_fail: bool = True # 失败是否重试

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "primitive": self.primitive,
            "params": self.params,
            "description": self.description,
            "timeout": self.timeout,
            "retry_on_fail": self.retry_on_fail,
        }


@dataclass
class TaskPlan:
    """规划模块输出: 完整任务计划"""
    task_id: str
    steps: list[TaskStep] = field(default_factory=list)
    estimated_duration: float = 0.0  # 预估耗时 (秒)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "steps": [s.to_dict() for s in self.steps],
            "estimated_duration": self.estimated_duration,
        }


@dataclass
class StepResult:
    """单步执行结果 — 执行模块反馈"""
    step_id: int
    status: TaskStatus = TaskStatus.PENDING
    message: str = ""
    elapsed: float = 0.0       # 实际耗时 (秒)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "message": self.message,
            "elapsed": self.elapsed,
        }


@dataclass
class ExecutionResult:
    """执行模块输出: 完整执行结果"""
    task_id: str
    overall_status: TaskStatus = TaskStatus.PENDING
    step_results: list[StepResult] = field(default_factory=list)
    error_message: str | None = None

    @property
    def is_success(self) -> bool:
        return self.overall_status == TaskStatus.SUCCESS

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "overall_status": self.overall_status.value,
            "step_results": [s.to_dict() for s in self.step_results],
            "error_message": self.error_message,
        }
