"""Common directory contract and local entry points for the four studies."""
import argparse
import contextlib
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .common import ROOT, PARTS, WORK_DIRS, read, save, sha, rms

STAGES = {
    1: ('verify',),
    2: ('verify', 'tests', 'refit'),
    3: ('verify', 'reproduce'),
    4: ('verify', 'tests', 'scan', 'analyze', 'refine', 'controls'),
}


def verify(number):
    """Validate retained evidence without rerunning or rewriting experiments."""
    base = ROOT / PARTS[number - 1]
    assert all((base / name).is_dir() for name in WORK_DIRS)
    assert (base / 'run.py').is_file() and (base / 'README.md').is_file()
    config = json.loads((base / 'config/experiment.json').read_text(encoding='utf8'))
    assert isinstance(config, dict) and config
    provenance = json.loads((ROOT / 'provenance.json').read_text(encoding='utf8'))
    retained = [r for r in provenance if Path(r['output']).parts[0] == base.name]
    assert retained
    for record in retained:
        assert sha(ROOT / record['output']) == record['output_sha256'], record['output']
    csv_rows = {}
    for path in sorted((base / 'data').glob('*.csv')):
        rows = read(path)
        assert rows, path
        assert all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v, float)), path
        csv_rows[path.name] = len(rows)
    if number == 1:
        assert csv_rows['bounded_range_scaling.csv'] == 16
        assert len({r['side'] for r in read(base / 'data/critical_crossover.csv')}) == 2
    elif number == 2:
        assert csv_rows['structural_validation.csv'] == 10
        tests = read(base / 'data/structural_checks.csv')
        assert len(tests) == 12 and max(r['error'] for r in tests) < 1e-7
        from .finite_range_refit import verify as verify_refit
        assert verify_refit()['status']=='passed'
    elif number == 3:
        rows = read(base / 'data/weak_field_results.csv')
        profiles = json.loads((base / 'profiles/weak_field_profiles.json').read_text(encoding='utf8'))
        assert len(rows) == len(profiles) == 288
        assert all(abs(rms(np.array(p['rho_plus']) - p['exact_plus']) - p['density_rms']) < 1e-20 for p in profiles)
        assert len(list((base / 'cache').glob('*nodes25.npz'))) == 12
    else:
        rows = read(base / 'data/window_scan.csv')
        assert len(rows) == len(list((base / 'profiles').glob('*.json'))) == 837
        assert csv_rows['applicability_windows.csv'] == 80
        assert csv_rows['conservative_windows.csv'] == 20
        for r in rows:
            p = json.loads((base / 'profiles' / f'{r["id"]}_{r["amplitude_kind"]}.json').read_text(encoding='utf8'))
            assert abs(rms(np.array(p['rho_NRC_LDA']) - p['rho_exact']) - r['error_rms']) < 1e-18
    return dict(status='passed',part=number,retained_hashes_checked=len(retained),csv_rows=csv_rows,
                mode='existing-evidence verification; no numerical experiment rerun')


def main(directory):
    base = Path(directory).resolve()
    if base.parent != ROOT or base.name not in PARTS:
        raise ValueError('Entry point must be one of the four TFIM study directories')
    number = PARTS.index(base.name) + 1
    parser = argparse.ArgumentParser(description='TFIM study workspace; default: verify existing evidence.')
    parser.add_argument('--stages', nargs='+', choices=STAGES[number], default=['verify'])
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest = dict(start_utc=stamp,part=number,stages=args.stages,completed=[],status='running')
    log = base / 'logs' / f'run_{stamp}'

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, text):
            for stream in self.streams: stream.write(text); stream.flush()
            return len(text)
        def flush(self):
            for stream in self.streams: stream.flush()

    try:
        with log.with_suffix('.txt').open('w', encoding='utf8') as output, contextlib.redirect_stdout(Tee(sys.stdout, output)), threadpool_limits(limits=1):
            for stage in args.stages:
                started = time.perf_counter()
                if stage == 'verify':
                    result = verify(number)
                    save(base / 'logs/verification.json', result)
                    print(f'PART {number} VERIFIED: {result["retained_hashes_checked"]} source hashes; {sum(result["csv_rows"].values())} CSV records', flush=True)
                elif stage == 'reproduce':
                    importlib.import_module('tfim.weak_field.__main__').main()
                elif stage == 'refit':
                    importlib.import_module('tfim.finite_range_refit').run()
                else:
                    module, function = {'tests':('study','tests'),'scan':('study','run'),'analyze':('analysis','run'),
                                        'refine':('refinement','run'),'controls':('controls','run')}[stage]
                    getattr(importlib.import_module('tfim.' + module), function)()
                manifest['completed'].append(dict(stage=stage,seconds=time.perf_counter()-started))
        manifest['status'] = 'passed'
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc))
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(log.with_suffix('.json'), manifest)


def documentation(number):
    commands = {
        1: '本部分保留解析结论与数值证据，入口核验来源哈希和数据结构；没有宣称重新生成历史临界核实验。',
        2: '`python run.py --stages tests refit verify` 重算 12 项结构检查与有限范围重拟合对照，并核验数据。重拟合配置在 `config/finite_range_refit.json`；第五轮原始文件已删除，当前对照由保留核表重新计算。',
        3: '`python run.py --stages reproduce verify` 独立重算 288 条弱场结果并比对；核输入复用 `cache/`，临时输出位于 `cache/replay/`，成功后清理。尾部消融的历史记录只核验来源哈希。',
        4: '`python run.py --stages scan analyze refine controls verify` 重算临界窗口实验。`tests` 可单独运行共享结构测试，结果归入第二部分；保留的高阶/多分支反例只核验来源哈希。',
    }
    return '''
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

''' + commands[number] + '''

共用算法在 `../tfim/`，不复制四份。没有对应产物的目录保留说明文件，不填充虚构结果。完整报告和跨项目审计从 TFIM 根目录运行 `python run.py --stages report audit`；该任务日志位于 `TFIM/logs/`。
'''


def directory_notes():
    (ROOT / 'DIRECTORY_STRUCTURE.md').write_text('''# TFIM 目录与运行入口

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
├── report/               本地生成的报告、逐表 CSV 和数据包，不提交 Git
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
''',encoding='utf8')
