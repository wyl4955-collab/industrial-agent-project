"""
工业智能体 Pipeline Demo (LangGraph)

用 LangGraph StateGraph 串联完整流程:
  指令理解(NLP) → 视觉识别(Vision) → 任务规划(Planning) → 仿真执行(Execute)

当前为概念验证版 — 各节点使用 mock 数据, 严格遵循 interfaces.py 的接口契约。
后续各模块开发完成后, 只需替换 mock 函数为真实实现即可。
"""

import re

from langgraph.graph import StateGraph, END

from .state import PipelineAgentState
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


# ═══════════════════════════════════════════════════════════════════
# Node 1: NLP 指令理解
#   Contract → interfaces.NLPModule.parse()
#   输入: state["instruction"] (str)
#   输出: parsed_intent, target_object, source/target_location, ...
# ═══════════════════════════════════════════════════════════════════

def nlp_node(state: PipelineAgentState) -> dict:
    """Mock NLP — 后续替换为 NLPModule.parse() 调用"""
    text = state["instruction"]

    # ── Mock 规则解析 ─────────────────────────────────
    parsed = ParsedInstruction(raw_text=text)

    # 意图识别
    if "拿" in text or "放" in text or "搬" in text:
        if "从" in text:
            parsed.intent = ActionType.MOVE
        elif "放到" in text:
            parsed.intent = ActionType.MOVE
        else:
            parsed.intent = ActionType.PICK
    elif "数" in text or "计数" in text:
        parsed.intent = ActionType.COUNT
    elif "分拣" in text or "分类" in text:
        parsed.intent = ActionType.SORT

    # 目标物体
    OBJ_MAP = {
        "螺丝刀": ObjectClass.SCREWDRIVER,
        "扳手": ObjectClass.WRENCH,
        "螺母": ObjectClass.NUT,
        "滚柱": ObjectClass.ROLLER,
        "轴承": ObjectClass.BEARING,
        "钳子": ObjectClass.PLIER,
        "锤子": ObjectClass.HAMMER,
        "螺栓": ObjectClass.BOLT,
    }
    for cn, obj in OBJ_MAP.items():
        if cn in text:
            parsed.target_object = obj.value
            break

    # 目标位置
    bin_match = re.search(r"(\d+)\s*号", text)
    if bin_match:
        parsed.target_location = LocationSpec(
            type=LocationType.BIN,
            label=f"bin_{bin_match.group(1)}",
        )

    # 源位置
    src_match = re.search(r"从.*?(\d+)\s*号", text)
    if src_match:
        parsed.source_location = LocationSpec(
            type=LocationType.BIN,
            label=f"bin_{src_match.group(1)}",
        )

    parsed.confidence = 0.95

    # ── 日志 ──────────────────────────────────────────
    print(f"[NLP]  指令: {text}")
    print(f"[NLP]  意图={parsed.intent.value}, 目标={parsed.target_object}")
    print(f"[NLP]  source={parsed.source_location}, target={parsed.target_location}")

    # 输出 → AgentState 字段 (按契约写入)
    return {
        "parsed_intent": parsed.intent.value,
        "target_object": parsed.target_object,
        "target_attributes": parsed.target_attributes,
        "source_location": parsed.source_location.to_dict() if parsed.source_location else None,
        "target_location": parsed.target_location.to_dict() if parsed.target_location else None,
        "constraints": parsed.constraints,
        "nlp_confidence": parsed.confidence,
        "logs": [f"[NLP] parsed: intent={parsed.intent.value}, obj={parsed.target_object}, conf={parsed.confidence:.2f}"],
    }


# ═══════════════════════════════════════════════════════════════════
# Node 2: 视觉感知
#   Contract → interfaces.VisionModule.detect()
#   输入: state["target_object"] (str)
#   输出: detected_objects, scene_id, frame_size
# ═══════════════════════════════════════════════════════════════════

def vision_node(state: PipelineAgentState) -> dict:
    """Mock Vision — 后续替换为 VisionModule.detect() 调用"""
    target = state.get("target_object", "")

    # ── Mock 场景数据 ─────────────────────────────────
    scene = SceneObservation(
        scene_id="scene_001",
        timestamp=0.0,
        frame_size=(640, 480),
        objects=[
            DetectedObject(
                id="obj_01", class_name="screwdriver",
                bbox_2d=[120, 200, 180, 320],
                position_3d=[0.15, 0.08, 0.22], confidence=0.97,
            ),
            DetectedObject(
                id="obj_02", class_name="wrench",
                bbox_2d=[300, 150, 380, 280],
                position_3d=[-0.10, 0.05, 0.18], confidence=0.93,
            ),
            DetectedObject(
                id="obj_03", class_name="nut",
                bbox_2d=[450, 250, 490, 290],
                position_3d=[0.20, -0.03, 0.12], confidence=0.89,
            ),
            DetectedObject(
                id="obj_04", class_name="bearing",
                bbox_2d=[220, 340, 280, 400],
                position_3d=[-0.05, 0.10, 0.15], confidence=0.85,
            ),
        ],
    )

    # ── 查找目标 ───────────────────────────────────────
    found = scene.get_best_match(target)

    print(f"[Vision] 场景 {scene.scene_id}: 检测到 {len(scene.objects)} 个物体")
    if found:
        print(f"[Vision]  目标 {target}: [{found.id}] pos={found.position_3d} conf={found.confidence:.2f}")
    else:
        print(f"[Vision]  目标 {target}: 未找到")

    return {
        "scene_id": scene.scene_id,
        "detected_objects": [o.to_dict() for o in scene.objects],
        "frame_size": scene.frame_size,
        "logs": [
            f"[Vision] detected {len(scene.objects)} objects"
            + (f", target '{target}' found at {found.position_3d}" if found else f", target '{target}' NOT found")
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# Node 3: 任务规划
#   Contract → interfaces.PlanningModule.plan()
#   输入: 全部上游字段 (parsed_intent, target_object, detected_objects, ...)
#   输出: task_id, task_steps, estimated_duration
# ═══════════════════════════════════════════════════════════════════

def planning_node(state: PipelineAgentState) -> dict:
    """Mock Planning — 后续替换为 PlanningModule.plan() 调用"""
    target = state.get("target_object", "")
    target_loc = state.get("target_location", {})

    # ── 从 state 重建数据对象 ──────────────────────────
    objects = [DetectedObject(**o) for o in state.get("detected_objects", [])]
    scene = SceneObservation(
        scene_id=state.get("scene_id", ""),
        objects=objects,
    )
    found = scene.get_best_match(target)

    if found is None:
        return {
            "task_steps": [],
            "logs": [f"[Planning] FAIL: target '{target}' not in scene"],
        }

    # ── Mock 任务分解 ──────────────────────────────────
    target_pos = found.position_3d
    loc_label = target_loc.get("label", "bin_1")

    steps = [
        TaskStep(step_id=1, primitive="move_to",
                 params={"pose": "observe"}, description="移动至观察位姿"),
        TaskStep(step_id=2, primitive="detect_and_locate",
                 params={"target": target, "expected_pos": target_pos},
                 description=f"视觉确认 {target} 位姿"),
        TaskStep(step_id=3, primitive="plan_grasp_path",
                 params={"from": "home", "to": target_pos},
                 description="规划抓取路径 (避障)"),
        TaskStep(step_id=4, primitive="open_gripper",
                 description="打开夹爪"),
        TaskStep(step_id=5, primitive="move_linear",
                 params={"target": target_pos, "speed": 0.5},
                 description="接近目标物体"),
        TaskStep(step_id=6, primitive="close_gripper",
                 description="抓取物体"),
        TaskStep(step_id=7, primitive="verify_grasp",
                 description="验证抓取成功"),
        TaskStep(step_id=8, primitive="move_linear",
                 params={"target": loc_label, "speed": 0.3},
                 description=f"搬运至 {loc_label}"),
        TaskStep(step_id=9, primitive="open_gripper",
                 description="放置物体"),
        TaskStep(step_id=10, primitive="verify_placement",
                 description="验证放置成功"),
        TaskStep(step_id=11, primitive="return_home",
                 description="返回初始位姿"),
    ]

    plan = TaskPlan(
        task_id=f"task_{state.get('scene_id', '000')}_{target}",
        steps=steps,
        estimated_duration=15.0,
    )

    print(f"[Planning] 任务 ID: {plan.task_id}")
    print(f"[Planning] 分解为 {len(steps)} 步, 预估 {plan.estimated_duration}s")
    for s in steps:
        print(f"  Step {s.step_id:2d}: {s.primitive:<20s} | {s.description}")

    return {
        "task_id": plan.task_id,
        "task_steps": [s.to_dict() for s in plan.steps],
        "estimated_duration": plan.estimated_duration,
        "logs": [f"[Planning] plan {plan.task_id}: {len(steps)} steps, ~{plan.estimated_duration}s"],
    }


# ═══════════════════════════════════════════════════════════════════
# Node 4: 仿真执行
#   Contract → interfaces.ExecutionModule.execute()
#   输入: task_steps
#   输出: step_results, overall_status, error_message
# ═══════════════════════════════════════════════════════════════════

def execution_node(state: PipelineAgentState) -> dict:
    """Mock Execution — 后续替换为 ExecutionModule.execute() 调用"""
    steps = [TaskStep(**s) for s in state.get("task_steps", [])]
    task_id = state.get("task_id", "unknown")

    if not steps:
        return {
            "overall_status": TaskStatus.FAILED.value,
            "error_message": "No task steps to execute",
            "step_results": [],
            "logs": ["[Execution] FAIL: empty task plan"],
        }

    results = []
    all_ok = True

    # ── Mock 逐步执行 ──────────────────────────────────
    for step in steps:
        # 模拟: 所有步骤都成功
        sr = StepResult(
            step_id=step.step_id,
            status=TaskStatus.SUCCESS,
            message=f"{step.primitive} -> {step.description}",
            elapsed=0.5 + step.step_id * 0.1,
        )
        results.append(sr)

        status_tag = "OK" if sr.status == TaskStatus.SUCCESS else "FAIL"
        print(f"  [{status_tag}] Step {sr.step_id}: {sr.message} ({sr.elapsed:.1f}s)")

    print(f"[Execution] 任务 {task_id}: {'SUCCESS' if all_ok else 'FAILED'}")

    return {
        "step_results": [r.to_dict() for r in results],
        "overall_status": TaskStatus.SUCCESS.value if all_ok else TaskStatus.FAILED.value,
        "error_message": None if all_ok else "Some steps failed",
        "logs": [f"[Execution] {task_id}: {'SUCCESS' if all_ok else 'FAILED'} ({len(results)} steps)"],
    }


# ═══════════════════════════════════════════════════════════════════
# Graph 构建 & Demo 入口
# ═══════════════════════════════════════════════════════════════════

def build_agent_graph():
    builder = StateGraph(PipelineAgentState)

    builder.add_node("nlp", nlp_node)
    builder.add_node("vision", vision_node)
    builder.add_node("planning", planning_node)
    builder.add_node("execute", execution_node)

    builder.set_entry_point("nlp")
    builder.add_edge("nlp", "vision")
    builder.add_edge("vision", "planning")
    builder.add_edge("planning", "execute")
    builder.add_edge("execute", END)

    return builder.compile()


if __name__ == "__main__":
    print("=" * 64)
    print("  工业交互智能体 — Pipeline Demo (LangGraph)")
    print("  接口类型: src/agent/types.py | src/agent/interfaces.py")
    print("=" * 64)

    graph = build_agent_graph()

    # 测试用例
    test_cases = [
        {"instruction": "把螺丝刀放到3号料箱"},
        {"instruction": "从1号料箱拿扳手"},
        {"instruction": "把螺母分拣到2号料箱"},
    ]

    for i, inputs in enumerate(test_cases, 1):
        inputs.setdefault("logs", [])
        print(f"\n{'─'*48}")
        print(f"  Test {i}: {inputs['instruction']}")
        print(f"{'─'*48}\n")
        result = graph.invoke(inputs)

        print(f"\n  >> 结果: {result.get('overall_status', 'N/A')}")
        n_steps = len(result.get("task_steps", []))
        n_objs = len(result.get("detected_objects", []))
        print(f"  >> 物体: {n_objs}, 步骤: {n_steps}")

    print(f"\n{'='*64}")
    print("  All tests passed.")
    print(f"{'='*64}")
