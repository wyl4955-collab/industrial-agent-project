# Industrial Agent Project

**工业环境下物体感知识别与指令交互型智能体**

基于 LangGraph 的端到端工业智能体框架，实现「环境感知 → 指令理解 → 任务分解 → 仿真执行」全流程自主运行。

**赛题编号**: XH-202607 | **发榜单位**: 上海电气集团中央研究院 | **截止日期**: 2026-09-05

---

## 架构

```
用户指令 (自然语言)
      │
  ┌───▼────┐     ┌─────────┐     ┌──────────┐     ┌────────┐
  │  NLP   │────▶│ Vision  │────▶│ Planning │────▶│Execute │
  │ 指令理解 │     │ 视觉感知  │     │ 任务分解  │     │ 仿真执行 │
  └────────┘     └─────────┘     └──────────┘     └────────┘
```

四个模块通过 LangGraph `StateGraph` 串联，共享 `PipelineAgentState` 上下文。

## 目录结构

```
├── src/
│   ├── agent/          # 系统集成 — LangGraph pipeline、接口契约
│   │   ├── types.py    # 枚举 + 数据类 (全模块共用)
│   │   ├── state.py    # PipelineAgentState
│   │   ├── interfaces.py  # 模块接口契约 (Protocol)
│   │   └── demo.py     # Pipeline 概念验证
│   ├── vision/         # 视觉感知 (潘铭凯、黄舒怡)
│   ├── nlp/            # 指令理解 (魏佳慧、吴若涵)
│   └── planning/       # 任务规划与仿真 (曾文博、王科蒙、郭爽)
├── docs/               # 接口规范、代码规范
├── tests/
└── requirements.txt
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Demo
python -m src.agent.demo
```

## 技术栈

| 模块 | 技术选型 |
|------|---------|
| 智能体框架 | LangChain + LangGraph |
| 视觉感知 | PyTorch + YOLOv8 |
| 指令理解 | BERT-base-chinese + 大模型 API 兜底 |
| 仿真执行 | PyBullet / CoppeliaSim |

## 队伍

| 分组 | 成员 |
|------|------|
| 项目总协调 | 雷妤 |
| 视觉感知 | 潘铭凯、黄舒怡 |
| NLP 指令 | 魏佳慧、吴若涵 |
| 任务规划与仿真 | 曾文博、王科蒙、郭爽 |
| 系统集成 | 刘文毅、郭爽 |
| 文档与项目管理 | 陈梓铫 |

## 提交物

- 可运行源代码 + 模型文件
- 仿真验证视频
- 技术报告
- 使用说明文档
