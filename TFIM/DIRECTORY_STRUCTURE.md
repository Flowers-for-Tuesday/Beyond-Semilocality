# TFIM 目录与运行入口

四个子项目采用相同的一级目录和入口。数值算法集中维护，不复制；实验数据按其科学用途归属到各子项目。

```text
TFIM/
├── 01_No-Go_Theorems/
├── 02_Minimal_Nonlocal_Correction/
├── 03_Error-Cancellation_Mechanism/
├── 04_Critical_Applicability_and_Limitations/
│   （每个子项目均有以下结构）
│   ├── README.md
│   ├── run.py
│   ├── config/experiment.json
│   ├── data/
│   ├── profiles/
│   ├── cache/
│   ├── figures/
│   └── logs/
├── tfim/                 共用数值代码、报告和审计代码
├── report/               四部分合并报告、逐表 CSV 和完整数据包
├── logs/                 跨项目运行日志
├── run.py                全流程入口
├── provenance.json       历史证据来源与当前文件哈希
├── layout_migration.json 本次文件位置映射与迁移前哈希
└── audit.json            全项目核验结果
```

## 各子项目支持的工作

下列命令在对应子项目目录中运行；`python run.py --help` 列出允许阶段。所有本地入口默认执行 `verify`，核验不会重算或覆盖科学数据。

| 部分 | 核验 | 可重算工作 |
| --- | --- | --- |
| 1 / No-Go | `python run.py` | 当前精简项目只保留解析结论、数值证据及其哈希核验 |
| 2 / Minimal correction | `python run.py` | `python run.py --stages tests refit verify`：12 项结构检查与有限范围重拟合对照；10 项历史检查保留核验 |
| 3 / Error cancellation | `python run.py` | `python run.py --stages reproduce verify`：288 条弱场结果独立复算；历史尾部消融保留核验 |
| 4 / Applicability | `python run.py` | `python run.py --stages scan analyze refine controls verify`：临界窗口全流程；历史反例保留核验 |

第四部分的 `tests` 调用共用结构检查并写入第二部分 `data/structural_checks.csv`。第四部分 `config/experiment.json` 是这些检查与临界扫描的数值配置；第三部分有自己独立的实验配置。

TFIM 根目录保持原来的全流程入口 `python run.py`。仅更新报告、来源索引和打包时运行 `python run.py --stages report audit`；跨项目任务日志写入根目录 `logs/`。第三部分原有 `python -m tfim.weak_field` 入口继续可用。

## 本次位置调整

第三部分的核输入从 `reproduction/cache/` 归入 `cache/`；独立复现记录归入 `logs/`，密度剖面归入 `profiles/`。第四部分的配置、波形与初值归入 `config/`，历史反例剖面归入 `profiles/retained/`。12 项新结构检查归入第二部分。原目录只在全部文件移动且逐字节哈希一致后移除。

未重新拟合参数或改动实验数值。`layout_migration.json` 保存全部 24 个移动文件的旧路径、新路径与 SHA-256；`provenance.json` 的当前路径已更新，历史来源字段保留原样。报告来源清单、PDF 和完整数据包随后重新生成。

后续新增：有限范围模型重新拟合对照归入第二部分，采用相同目录结构。该对照由保留核表重建；第五轮原始文件未恢复。可用该部分的 `refit` 阶段重新计算。
