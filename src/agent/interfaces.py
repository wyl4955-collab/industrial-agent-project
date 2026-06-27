"""
模块接口契约 — 系统集成组维护

本文档定义了三大模块的输入/输出规范。各组实现时:
  - 函数签名必须与下方完全一致
  - 输入数据从 PipelineAgentState 中读取对应字段
  - 输出数据按契约返回对应字段的 dict
  - 模块内部实现完全自由, 集成组只关心契约

数据流:
  用户指令 (str)
      │
      ▼
  ┌─────────────┐
  │  NLP 模块    │  输入: instruction → 输出: parsed_intent, target_object, ...
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Vision 模块 │  输入: target_object → 输出: detected_objects, scene_id
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Planning 模块│ 输入: 全部上游字段 → 输出: task_steps
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Execute 模块 │ 输入: task_steps → 输出: step_results, overall_status
  └──────────────┘
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

from .types import (
    DetectedObject,
    ExecutionResult,
    ParsedInstruction,
    SceneObservation,
    StepResult,
    TaskPlan,
    TaskStatus,
)


# ═══════════════════════════════════════════════════════════════════
# 1. NLP 指令理解模块接口
# ═══════════════════════════════════════════════════════════════════

class NLPModule(Protocol):
    """NLP 模块契约

    由 NLP 组 (魏佳慧、吴若涵) 实现。支持两种模式:
      模式 A (轻量本地): BERT-base-chinese 做实体抽取, 覆盖 80% 常规指令
      模式 B (大模型兜底): 通义千问/文心一言 API 处理复杂/歧义指令
      模式 C (微调加分): 工业指令领域微调 BERT, 提升专业术语准确率
        (赛题明确: 自行微调模型 = 技术先进性 +15分)

    LangGraph 节点包装 (集成组负责):
      def nlp_node(state: PipelineAgentState) -> dict:
          instruction = state["instruction"]
          parsed = nlp_module.parse(instruction)
          return parsed.to_dict()

    测试用例:
      "把螺丝刀放到3号料箱"       → intent=move, target=screwdriver, target=bin_3
      "从第二个格子拿扳手"         → intent=pick, target=wrench, source=bin_2
      "将红色螺母放到左边料箱"     → intent=move, target=nut, attributes={color:red}
      "数一下有多少个滚柱"         → intent=count, target=roller
    """

    @abstractmethod
    def parse(self, instruction: str) -> ParsedInstruction:
        """解析自然语言指令 → 结构化指令

        Args:
            instruction: 原始中文指令文本

        Returns:
            ParsedInstruction: 包含意图/目标物体/位置/约束的结构化结果

        Raises:
            NLPParseError: 无法解析时抛出, 集成组会降级调用大模型
        """
        ...

    @abstractmethod
    def parse_with_llm(self, instruction: str) -> ParsedInstruction:
        """大模型兜底解析 (处理复杂/歧义指令)"""
        ...

    @abstractmethod
    def load_finetuned_model(self, model_path: str) -> bool:
        """加载工业指令领域微调模型 (加分项)

        Args:
            model_path: 微调后的 BERT 模型路径

        Returns:
            True 表示加载成功, 后续 parse() 自动走微调模型
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# 2. 视觉感知模块接口
# ═══════════════════════════════════════════════════════════════════

class VisionModule(Protocol):
    """视觉模块契约

    由视觉组 (潘铭凯、黄舒怡) 实现。基于 YOLOv8 做目标检测+实例分割。
    输出每个物体的 2D 像素坐标 + 3D 世界坐标。

    加分项: 针对工业工具场景微调 YOLO, 提升小样本/遮挡/混杂堆叠识别精度
      (赛题明确: 自行微调模型 = 技术先进性 +15分)

    LangGraph 节点包装:
      def vision_node(state: PipelineAgentState) -> dict:
          scene = vision_module.detect(target_class=state["target_object"])
          return {
              "detected_objects": [o.to_dict() for o in scene.objects],
              "scene_id": scene.scene_id,
              "frame_size": scene.frame_size,
          }

    测试场景:
      桌面场景含 5 种工具混合摆放 (螺丝刀/扳手/螺母/滚柱/钳子)
      部分遮挡 30%, 光照变化±20%
      检测每帧 < 100ms, mAP >= 90%
    """

    @abstractmethod
    def detect(
        self, image=None, target_class: str = "", top_k: int = 5
    ) -> SceneObservation:
        """检测场景中的所有物体

        Args:
            image: 输入图像 (numpy array, BGR 格式), None 表示使用仿真场景渲染
            target_class: 目标类别筛选 (空字符串=返回全部)
            top_k: 最多返回前 K 个结果

        Returns:
            SceneObservation: 包含所有检测物体的列表及场景元信息
        """
        ...

    @abstractmethod
    def get_target_position(self, target_class: str) -> DetectedObject | None:
        """直接获取目标物体的 3D 坐标 (便捷方法)"""
        ...

    @abstractmethod
    def load_finetuned_model(self, model_path: str) -> bool:
        """加载工业工具场景微调模型 (加分项)

        Args:
            model_path: 微调后的 YOLO 权重路径 (.pt 文件)

        Returns:
            True 表示加载成功, 后续 detect() 自动走微调模型
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# 3. 任务规划模块接口
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# 原子动作原语 — 规划组/执行组共用
# ═══════════════════════════════════════════════════════════════════

PRIMITIVES: dict[str, str] = {
    # 机械臂运动
    "move_to": "移动到指定关节位姿 / 笛卡尔位姿",
    "move_linear": "直线路径移动 (避免碰撞物体)",
    "return_home": "返回机械臂初始位置",
    # 夹爪控制
    "open_gripper": "打开夹爪 (准备抓取)",
    "close_gripper": "关闭夹爪 (执行抓取)",
    # 感知动作
    "detect_and_locate": "调用视觉模块确认目标位姿",
    "verify_grasp": "抓取后验证 (检查是否成功拿起)",
    "verify_placement": "放置后验证 (检查是否在目标位置)",
    # 等待/延迟
    "wait": "等待指定时间",
}

# 料箱世界坐标映射 — 仿真组统一配置, 视觉组/规划组共用
BIN_POSITIONS: dict[str, list[float]] = {
    "bin_1": [0.30, -0.15, 0.05],
    "bin_2": [0.30, -0.05, 0.05],
    "bin_3": [0.30, 0.05, 0.05],
    "bin_4": [0.30, 0.15, 0.05],
    "bin_5": [0.30, -0.10, 0.15],
    "bin_6": [0.30, 0.10, 0.15],
    "table": [0.00, 0.00, 0.02],
    "home": [0.00, -0.30, 0.30],
}


class PlanningModule(Protocol):
    """规划模块契约

    由仿真规划组 (曾文博、王科蒙、郭爽) 实现。
    输入 NLP 解析结果 + 视觉检测结果, 输出可执行的原子任务序列。

    LangGraph 节点包装:
      def planning_node(state: PipelineAgentState) -> dict:
          # 1. 重建 ParsedInstruction
          parsed = ParsedInstruction(
              raw_text=state["instruction"],
              intent=ActionType(state["parsed_intent"]),
              target_object=state["target_object"],
              ...
          )
          # 2. 重建 SceneObservation
          scene = SceneObservation(
              scene_id=state.get("scene_id", ""),
              objects=[DetectedObject(**o) for o in state["detected_objects"]],
          )
          # 3. 规划
          plan = planning_module.plan(parsed, scene)
          return {"task_steps": [s.to_dict() for s in plan.steps]}

    测试用例:
      简单取放: "把螺丝刀放到3号料箱"
      多步搬运: "把扳手从1号料箱拿到5号料箱"
      分拣: "把所有螺母分拣到2号料箱"
      失败重试: 抓取失败后自动调整位姿重试
    """

    @abstractmethod
    def plan(self, instruction: ParsedInstruction, scene: SceneObservation) -> TaskPlan:
        """根据指令和场景生成任务计划

        Args:
            instruction: NLP 模块解析后的结构化指令
            scene: 视觉模块输出的场景观测

        Returns:
            TaskPlan: 原子动作序列和预估耗时

        Raises:
            PlanningError: 无法生成有效计划 (物体缺失/位置不可达等)
        """
        ...

    @abstractmethod
    def replan_on_failure(
        self,
        original_plan: TaskPlan,
        failed_step: StepResult,
        scene: SceneObservation,
    ) -> TaskPlan:
        """执行失败时重新规划 (含重试策略)"""
        ...


# ═══════════════════════════════════════════════════════════════════
# 4. 仿真执行模块接口
# ═══════════════════════════════════════════════════════════════════

class ExecutionModule(Protocol):
    """执行模块契约

    由仿真规划组 (曾文博、王科蒙、郭爽) 实现。
    逐步执行 TaskPlan, 返回每步执行结果。

    LangGraph 节点包装:
      def execution_node(state: PipelineAgentState) -> dict:
          plan = TaskPlan(
              task_id=state["task_id"],
              steps=[TaskStep(**s) for s in state["task_steps"]],
          )
          result = executor.execute(plan)
          return {
              "step_results": [s.to_dict() for s in result.step_results],
              "overall_status": result.overall_status.value,
              "error_message": result.error_message,
          }
    """

    @abstractmethod
    def execute(self, plan: TaskPlan, step_callback=None) -> ExecutionResult:
        """逐步执行任务计划

        Args:
            plan: 待执行的任务计划
            step_callback: 每步执行后的回调, 用于实时日志/UI 更新

        Returns:
            ExecutionResult: 包含每步状态的完整执行结果
        """
        ...

    @abstractmethod
    def execute_single_step(self, step: TaskStep) -> StepResult:
        """执行单步原子动作 (用于调试和手动控制)"""
        ...
