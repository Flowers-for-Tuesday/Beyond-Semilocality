# 2 / Minimal Nonlocal Correction

| 验证 | 目的 | 简洁结论 |
| --- | --- | --- |
| M1：核差构造 | 固定精确均匀 LDA 后识别唯一缺失的二阶信息 | ΔK=Kexact(q;r)−Kexact(0;r) 由完整 Hessian 要求唯一确定 |
| M2：变分与对称性 | 核对完整 RMS 导数、均匀 Hessian 与局域符号不变性 | SARC 是变分一致的一项非局域修复 |
| M3：有限范围重新拟合 | 排除固定范围模型只是系数未调好的解释 | 重拟合降低核误差，但 R=8 的密度误差仍为一阶；SARC 为二阶 |

主泛函：F=Σ fhom(ρi)+(8Nr²)⁻¹Σq≠0 ΔK(q;r)|uq|²，u=ρ²，r²=〈ρ²〉。不新增拟合系数。平方坐标适配对称性，RMS 是选定的延拓约定；唯一性只到二阶核，不到完整非线性泛函。

`data/structural_validation.csv` 保留直接相关的 10 项既有检查；`data/structural_checks.csv` 保存 12 项广域核与热力学求解器检查。

重新拟合对照保留 R=1/2/4/8 的全部 24 组核结果；R=8 另做 24 组波形、四个幅度，共 96 条弱场验证。系数按最大相对核误差拟合，不加系数约束；验证动量与拟合动量分开。密度误差指数为 0.9933–1.0074，仍有领先线性项。SARC 用一项完整核差消去该项，体现最小必要非局域修复的作用。

第五轮原始重拟合文件已在此前清理中删除；当前对照是根据保留精确核与 JW 密度重建的实验。设置见 `config/finite_range_refit.json`，标量结果见 `data/finite_range_refit_*.csv`，96 条密度剖面见 `profiles/finite_range_refit_profiles.json`，来源哈希与运行记录见 `logs/finite_range_refit.json`。

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

`python run.py --stages tests refit verify` 重算 12 项结构检查与有限范围重拟合对照，并核验数据。重拟合配置在 `config/finite_range_refit.json`；第五轮原始文件已删除，当前对照由保留核表重新计算。

共用算法在 `../tfim/`，不复制四份。没有对应产物的目录保留说明文件，不填充虚构结果。完整报告和跨项目审计从 TFIM 根目录运行 `python run.py --stages report audit`；该任务日志位于 `TFIM/logs/`。
