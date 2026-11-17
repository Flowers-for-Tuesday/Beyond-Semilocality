# TFIM：最小必要非局域修复

从两条 No-Go 出发，以一个 SARC 非局域修正补回缺失均匀逆响应核；验证误差消除，并量化临界逼近时残余非线性误差的可用窗口。

[完整 PDF 报告（含重拟合对照）](report/TFIM_核心研究与临界适用窗口_含重拟合对照.pdf) · [可编辑正文](report/TFIM_report.md) · [目录与运行入口](DIRECTORY_STRUCTURE.md) · [来源清单](provenance.json) · [剔除清单](selection.json)

## 1 / No-Go Theorems

[必要性：临界非解析核与交叉函数](01_No-Go_Theorems/README.md)。只保留支持两条定理的证据。

## 2 / Minimal Nonlocal Correction

[最小二阶核差、完整变分与有限范围重拟合对照](02_Minimal_Nonlocal_Correction/README.md)。重拟合有限范围系数可降低误差，但仍保留一阶密度误差；SARC 用一项完整核差消除该项。

## 3 / Error-Cancellation Mechanism

[24 组弱场与纯尾部消融](03_Error-Cancellation_Mechanism/README.md)。保留 288 条关键结果及独立复现代码，去除重复对照和候选模型实验。

## 4 / Critical Applicability and Limitations

[新增 620 个主网格外场与 217 个边界加密点的适用窗口](04_Critical_Applicability_and_Limitations/README.md)。以误差占诱导密度响应的 1%/5% 定义“可以忽略”，同时检验临界距离、扰动幅度和波长。

固定 L=32、λ=0.001，最坏相对误差由 m=0.1 时约 0.0855% 增至 m=0.0001 时约 6.479%。**消除领先线性误差，不保证固定幅度下无限逼近临界点仍然准确。**

## 复现与目录边界

在本目录运行：

```powershell
python run.py
```

默认依次执行结构测试、新临界扫描、窗口提取、阈值边界加密、边界数值控制、报告和审计。也可使用 `python run.py --stages analyze report audit`。第三部分全部 288 条保留弱场结果可用 `python -m tfim.weak_field` 独立重算；第四部分可用默认命令重算。原始密度、关键反例记录和必要核输入均已迁入 TFIM，不依赖任何旧目录。

四个子项目均使用 `config / data / profiles / cache / figures / logs`，各有 `README.md` 和 `run.py`。进入任一子项目运行 `python run.py` 默认核验已有数据；可重算阶段见 [目录与运行入口](DIRECTORY_STRUCTURE.md)。第三、四部分的实验配置分别位于各自的 `config/experiment.json`，跨项目任务日志在根目录 `logs/`。

`provenance.json` 给出保留证据与来源哈希；`audit.json` 检查数据、公式对应的观测量、窗口分类和报告。PDF 需要 Typst 及 SimSun/Cambria/Cambria Math，审计需要 PyMuPDF；Python 依赖见 `requirements.txt`。

TFIM 是唯一研究项目。原 `tfim_critical_dft` 目录与 PROMPT1–5 探索文档已删除。保留有限范围重新拟合，作为 SARC 最小必要性的对照；该实验由现存核表重建，未恢复第五轮原文件。PLR/H2/L2、与结构验证无关的高阶拟合和重复排名不保留。`provenance.json` 中的历史路径用于溯源；重拟合的输入哈希另记在第二部分运行日志中。
