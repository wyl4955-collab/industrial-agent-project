# 代码规范与协作约定

**维护者**: 系统集成组 (刘文毅)  
**更新日期**: 2026-06-27

---

## 一、命名规则

### 文件与目录
| 规则 | 示例 |
|------|------|
| Python 文件: `snake_case` | `interface_spec.py`, `vision_detector.py` |
| 目录: `snake_case` | `src/agent/`, `src/vision/` |
| 测试文件: `test_` 前缀 | `tests/test_nlp_parse.py` |

### 代码
| 规则 | 示例 |
|------|------|
| 类名: `PascalCase` | `ParsedInstruction`, `DetectedObject` |
| 函数/方法: `snake_case` | `detect()`, `get_best_match()` |
| 变量: `snake_case` | `target_object`, `scene_id` |
| 常量: `UPPER_SNAKE` | `BIN_POSITIONS`, `PRIMITIVES` |
| 枚举值: `UPPER_SNAKE` | `ActionType.PICK`, `TaskStatus.SUCCESS` |
| 私有成员: `_` 前缀 | `_internal_cache` |

### 字段命名与 codegen 一致性
`types.py` 中 dataclass 的字段名 → `to_dict()` 输出的 key 名称必须一致。State 中存的是 dict，重建 dataclass 时用 `Dataclass(**dict)` 展开。

---

## 二、目录结构

```
industrial-agent-project/
├── src/
│   ├── agent/          # 系统集成组 — LangGraph pipeline、接口类型
│   │   ├── types.py    #   枚举 + 数据类 (所有模块共用)
│   │   ├── state.py    #   PipelineAgentState (LangGraph TypedDict)
│   │   ├── interfaces.py  #   模块接口契约 (Protocol)
│   │   └── demo.py     #   Pipeline 概念验证 Demo
│   ├── vision/         # 视觉感知组 — YOLOv8 检测与坐标输出
│   ├── nlp/            # NLP指令组 — 指令解析与实体抽取
│   └── planning/       # 仿真规划组 — 仿真场景、任务分解、执行
├── tests/              # 所有测试代码
├── docs/               # 项目文档
└── requirements.txt    # Python 依赖清单
```

各组在自己的目录下自由组织代码，对外暴露的入口函数签名必须符合 `src/agent/interfaces.py` 中的契约。

---

## 三、导入规则

```python
# 跨模块导入使用绝对路径
from src.agent.types import DetectedObject, SceneObservation
from src.agent.interfaces import BIN_POSITIONS

# 同模块内使用相对路径
from .types import ParsedInstruction
```

---

## 四、Git 协作约定

### 分支管理
- `main` — 保护分支，只接受 PR 合并
- 各组开发分支: `vision/<feature>`, `nlp/<feature>`, `planning/<feature>`, `integration/<feature>`

### Commit 格式
```
<type>: <简短描述>

- <变更点1>
- <变更点2>
```

type 取值: `feat` (新功能), `fix` (修复), `docs` (文档), `refactor` (重构), `test` (测试)

### 示例
```
feat: 实现 YOLOv8 目标检测模块

- 支持 8 类工业工具检测
- 输出 2D bbox + 3D 世界坐标
- mAP@0.5 达到 92%
```

---

## 五、代码风格

- **编码**: UTF-8
- **缩进**: 4 空格 (不用 Tab)
- **行宽**: 不超过 100 字符
- **字符串**: 优先双引号 `"..."`, f-string 用于插值
- **类型注解**: 所有公共函数必须标注参数和返回值类型
- **docstring**: 公共函数用 triple-quote 简短说明，一行即可
