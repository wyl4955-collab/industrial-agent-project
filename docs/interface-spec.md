# 模块接口规范 v1.0

**维护者**: 系统集成组 (刘文毅)
**更新日期**: 2026-06-28

---

## 附: 赛题评审标准

| 评审维度                         | 分值         | 关键点                                        |
| -------------------------------- | ------------ | --------------------------------------------- |
| 功能完整性 — 全流程             | 20           | 感知→识别→执行完整闭环                      |
| 功能完整性 — 模块可迁移         | 15           | 各模块独立替换, 框架通用                      |
| 技术先进性 — 感知精度           | 10           | mAP/推理速度达标                              |
| 技术先进性 — 任务分解           | 10           | 序列合理性, 执行成功率                        |
| **技术先进性 — 微调效果** | **15** | **自行微调工业场景模型 (最大单项加分)** |
| 系统工程化 — 代码结构           | 5            | 注释清晰, 可复现                              |
| 系统工程化 — 仿真/实物          | 5+10         | 仿真验证 5分, 真实机械臂 10分                 |
| 文档与展示                       | 10           | 报告逻辑 5分, 视频 5分                        |


## 一、数据流全景

```
用户自然语言指令 (str)
        │
        ▼
┌──────────────────┐
│  NLP 指令理解模块  │  魏佳慧、吴若涵
│  parse(text)→PI   │  输出: ParsedInstruction
└────────┬─────────┘
         │  target_object, intent, location, ...
         ▼
┌──────────────────┐
│  视觉感知模块      │  潘铭凯、黄舒怡
│  detect()→SO      │  输出: SceneObservation
└────────┬─────────┘
         │  detected_objects (with 3D coords)
         ▼
┌──────────────────┐
│  任务规划模块      │  曾文博、王科蒙、郭爽
│  plan(PI,SO)→TP   │  输出: TaskPlan
└────────┬─────────┘
         │  task_steps (原子动作序列)
         ▼
┌──────────────────┐
│  仿真执行模块      │  曾文博、王科蒙、郭爽
│  execute(TP)→ER   │  输出: ExecutionResult
└──────────────────┘
```

**坐标系约定**:

- 世界坐标系: 桌面中心原点, X=前, Y=左, Z=上 (米)
- 像素坐标系: 左上角原点, X=右, Y=下 (像素)

---

## 二、共用枚举

### ActionType (动作意图) — NLP模块输出

| 值          | 含义             | 示例指令               |
| ----------- | ---------------- | ---------------------- |
| `pick`    | 抓取             | "把螺丝刀拿起来"       |
| `place`   | 放置             | "放到3号料箱"          |
| `move`    | 搬运(pick+place) | "把扳手从1号拿到3号"   |
| `sort`    | 分拣             | "把所有螺母分拣出来"   |
| `count`   | 计数             | "数一下有几个滚柱"     |
| `inspect` | 检查             | "检查轴承是否在料箱里" |

### ObjectClass (物体类别) — NLP & Vision 共用

`screwdriver` | `wrench` | `nut` | `bearing` | `roller` | `plier` | `hammer` | `bolt` | `unknown`

### TaskStatus (步骤状态)

`pending` → `running` → `success` / `failed` / `skipped`

---

## 三、NLP 模块接口

### 函数签名

```python
def parse(instruction: str) -> ParsedInstruction
def parse_with_llm(instruction: str) -> ParsedInstruction  # 大模型兜底
```

### 输出字段: ParsedInstruction

| 字段                  | 类型                  | 必填 | 说明                            |
| --------------------- | --------------------- | ---- | ------------------------------- |
| `raw_text`          | `str`               | ✓   | 原始指令文本                    |
| `intent`            | `ActionType`        | ✓   | 动作意图                        |
| `target_object`     | `str`               | ✓   | 目标类别 (ObjectClass值)        |
| `target_attributes` | `dict`              |      | 属性约束, 如`{"color":"red"}` |
| `source_location`   | `LocationSpec\|None` |      | 抓取来源                        |
| `target_location`   | `LocationSpec\|None` |      | 放置目标                        |
| `constraints`       | `list[str]`         |      | 额外约束                        |
| `confidence`        | `float`             | ✓   | 置信度 [0,1]                    |

### LocationSpec

```python
{"type": "bin", "label": "bin_3", "coord": null}     # 料箱
{"type": "table", "label": "table", "coord": null}   # 桌面
{"type": "coord", "label": "", "coord": [x,y,z]}     # 显式坐标
```

### 验收用例

```
"把螺丝刀放到3号料箱"      → intent=move, obj=screwdriver, target=bin_3
"从第二个格子拿扳手"        → intent=pick, obj=wrench, source=bin_2
"将红色螺母放到左边料箱"    → intent=move, obj=nut, attrs={color:red}
"数一下有几个滚柱"          → intent=count, obj=roller
```

---

## 四、视觉模块接口

### 函数签名

```python
def detect(image=None, target_class: str = "", top_k: int = 5) -> SceneObservation
def get_target_position(target_class: str) -> DetectedObject | None
```

### 输出字段: SceneObservation

| 字段           | 类型                     | 说明                   |
| -------------- | ------------------------ | ---------------------- |
| `scene_id`   | `str`                  | 场景标识 (帧号/时间戳) |
| `objects`    | `list[DetectedObject]` | 所有检测到的物体       |
| `timestamp`  | `float`                | 采集/推理时刻          |
| `frame_size` | `(w, h)`               | 图像分辨率             |

### DetectedObject (每个物体)

| 字段            | 类型              | 说明                                 |
| --------------- | ----------------- | ------------------------------------ |
| `id`          | `str`           | 唯一标识                             |
| `class_name`  | `str`           | 类别 (ObjectClass值)                 |
| `bbox_2d`     | `[x1,y1,x2,y2]` | 像素框                               |
| `position_3d` | `[x,y,z]`       | **世界坐标**(米) — 最关键字段 |
| `confidence`  | `float`         | 置信度 [0,1]                         |
| `orientation` | `[r,p,y]\|None`  | 姿态角 (可选)                        |
| `dimensions`  | `[w,h,d]\|None`  | 尺寸 (可选)                          |

### 验收指标

- 检测目标: 5种以上工业工具混合摆放
- mAP >= 90%, 单帧推理 <= 100ms
- 坐标误差: 2D bbox IoU >= 0.85, 3D 位置误差 <= 2cm

---

## 五、任务规划模块接口

### 函数签名

```python
def plan(instruction: ParsedInstruction, scene: SceneObservation) -> TaskPlan
def replan_on_failure(plan: TaskPlan, failed_step: StepResult, scene: SceneObservation) -> TaskPlan
```

### 输出字段: TaskPlan

| 字段                   | 类型               | 说明         |
| ---------------------- | ------------------ | ------------ |
| `task_id`            | `str`            | 任务唯一ID   |
| `steps`              | `list[TaskStep]` | 原子动作序列 |
| `estimated_duration` | `float`          | 预估耗时(秒) |

### TaskStep (每步原子动作)

| 字段              | 类型      | 说明                          |
| ----------------- | --------- | ----------------------------- |
| `step_id`       | `int`   | 步骤序号 (从1递增)            |
| `primitive`     | `str`   | **原子原语名** (见下表) |
| `params`        | `dict`  | 原语参数                      |
| `description`   | `str`   | 人类可读描述                  |
| `timeout`       | `float` | 超时(秒), 默认5.0             |
| `retry_on_fail` | `bool`  | 失败是否重试, 默认True        |

### 原子原语词汇表 (定义在 `interfaces.py` 模块级 `PRIMITIVES` 字典)

| 原语                  | 参数                                   | 说明             |
| --------------------- | -------------------------------------- | ---------------- |
| `move_to`           | `pose: str`                          | 移至指定位姿     |
| `move_linear`       | `target: str\|list`, `speed: float` | 直线运动         |
| `open_gripper`      | —                                     | 开夹爪           |
| `close_gripper`     | —                                     | 合夹爪           |
| `detect_and_locate` | `target: str`                        | 调用视觉确认位置 |
| `plan_grasp_path`   | `from: str`, `to: list`            | 规划避障路径     |
| `verify_grasp`      | —                                     | 验证是否抓稳     |
| `verify_placement`  | —                                     | 验证是否正确放置 |
| `return_home`       | —                                     | 回初始位姿       |
| `wait`              | `duration: float`                    | 等待             |

### 料箱世界坐标 (定义在 `interfaces.py` 模块级 `BIN_POSITIONS` 字典)

```
bin_1: [0.30, -0.15, 0.05]    bin_4: [0.30,  0.15, 0.05]
bin_2: [0.30, -0.05, 0.05]    bin_5: [0.30, -0.10, 0.15]
bin_3: [0.30,  0.05, 0.05]    bin_6: [0.30,  0.10, 0.15]
table: [0.00,  0.00, 0.02]    home:  [0.00, -0.30, 0.30]
```

### 验收用例

```
简单取放: pick + place 完整流程
多步搬运: move with different source/target
分拣: sort + 多轮循环
失败重试: 故意抓取失败 → 自动调整 → 重试成功
```

---

## 六、执行模块接口

### 函数签名

```python
def execute(plan: TaskPlan, step_callback=None) -> ExecutionResult
def execute_single_step(step: TaskStep) -> StepResult
```

### 输出字段: ExecutionResult

| 字段               | 类型                 | 说明         |
| ------------------ | -------------------- | ------------ |
| `task_id`        | `str`              | 任务ID       |
| `overall_status` | `TaskStatus`       | 整体状态     |
| `step_results`   | `list[StepResult]` | 每步执行结果 |
| `error_message`  | `str\|None`         | 错误信息     |

### StepResult (每步反馈)

| 字段        | 类型           | 说明                   |
| ----------- | -------------- | ---------------------- |
| `step_id` | `int`        | 对应步骤ID             |
| `status`  | `TaskStatus` | success/failed/skipped |
| `message` | `str`        | 执行描述               |
| `elapsed` | `float`      | 实际耗时(秒)           |

---

## 七、各组对接步骤

1. **clone 仓库** → `git clone https://github.com/wyl4955-collab/industrial-agent-project.git`
2. **阅读规范文档** → `docs/code-standards.md` (命名规则、目录结构、Git 协作约定)
3. **阅读接口文件** → `src/agent/types.py` (数据结构), `src/agent/interfaces.py` (契约)
4. **在对应目录下开发**:
   - NLP组 → `src/nlp/`
   - 视觉组 → `src/vision/`
   - 规划+执行 → `src/planning/`
5. **函数签名必须匹配**: 集成组走的 LangGraph 节点需要调你们的函数, 签名不对就接不上
6. **输出必须调 `.to_dict()`**: 所有数据类型都自带 `to_dict()` 方法, State 中存 dict

## 八、常见问题

**Q: 我的模块需要额外的输入怎么办?**
A: 如果是全局配置 (如相机内参、机械臂关节角), 通过 constructor 传入你的类实例, 不要往 AgentState 加字段。如果确实需要跨模块共享的新字段, 找集成组统一加。

**Q: 模块间如何调试?**
A: 每个模块可以独立单元测试。集成组会在第4周提供 Mock 上下游, 你可以拿着 Mock 数据测试自己的模块。

**Q: 字段名/枚举值不满意?**
A: 直接跟集成组 (刘文毅) 说, 接口规范是活的, 但现在不改以后就越来越难改了。
