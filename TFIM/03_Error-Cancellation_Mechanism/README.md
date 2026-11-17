# 3 / Error-Cancellation Mechanism

| 实验 | 目的 | 简洁结论 |
| --- | --- | --- |
| E1：24 组多模弱场 | 用独立非均匀 JW 解验证消除线性项 | SARC 密度误差指数 1.99861–2.00190；Euler 缺陷指数 1.99898–2.00093 |
| E2：尾部消融 | 区分核结构修复与一般精度提升 | R8 截断恢复一阶误差；其系数由均匀核准确预测 |

只保留 LDA、SARC-R8、SARC，共 288 条结果与完整正负场剖面；删除重复 GGA 对照。保留 54 行既有尾部因果证据，包括未分辨标记。已有单模、双模重复验证不再单独复制。

在 TFIM 根目录运行 `python -m tfim.weak_field` 可独立重算全部 288 条弱场结果，并与保留数据核对。配置在 `config/experiment.json`，必要核输入在 `cache/`，完整密度在 `profiles/weak_field_profiles.json`。临时重算输出在核验后清理，只保留 `logs/reproduction_check.json`；不依赖已删除的旧项目。

同一近似泛函内部的松弛差为四阶，仅作为误差机制的补充；不能称真实基态能量达到四阶精度。R8 仍含全局 RMS，只是均匀 Hessian 有限范围。

## 统一工作目录

```text
README.md                本部分的目的、结论与运行说明
run.py                   本部分的统一入口，默认只核验已有数据
config/experiment.json   实验配置或保留证据的核验范围
data/                    标量结果、拟合、窗口和汇总表
profiles/                完整密度剖面；历史反例位于 retained/
cache/                   数值核输入、精确解缓存与临时重算目录
figures/                 本部分论文图（PNG/PDF）
logs/                    运行日志、数据核验和独立复现记录
```

在本目录运行 `python run.py` 或 `python run.py --stages verify`；从其他目录运行本目录 `run.py` 的绝对路径也可。使用 `python run.py --help` 查看支持的阶段。

`python run.py --stages reproduce verify` 独立重算 288 条弱场结果并比对；核输入复用 `cache/`，临时输出位于 `cache/replay/`，成功后清理。尾部消融的历史记录只核验来源哈希。

共用算法在 `../tfim/`，不复制四份。没有对应产物的目录保留说明文件，不填充虚构结果。完整报告和跨项目审计从 TFIM 根目录运行 `python run.py --stages report audit`；该任务日志位于 `TFIM/logs/`。
