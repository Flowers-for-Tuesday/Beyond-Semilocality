import json,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .common import ROOT,NEW,PARTS,read,write,save

COL={'LDA':'#7a818b','SARC-R8':'#a34f79','SARC':'#087e91'}
def data(part,name):return read(ROOT/PARTS[part]/'data'/f'{name}.csv')
def finish(fig,name):
    part=0 if name.startswith('01_') else 2 if name.startswith('02_') else 3
    folder=ROOT/PARTS[part]/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
def edge(a):
    a=np.log(np.array(a));return np.exp(np.r_[a[0]-(a[1]-a[0])/2,(a[1:]+a[:-1])/2,a[-1]+(a[-1]-a[-2])/2])
def figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'legend.frameon':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,3,figsize=(10.4,2.9),layout='constrained');rr=sorted(data(0,'critical_kernel'),key=lambda r:r['q']);q=np.array([r['q'] for r in rr]);k=np.array([r['K_exact'] for r in rr]);mask=q<.1;axs[0].semilogx(q[mask],k[mask]*np.log(1/q[mask])/np.pi,color=COL['SARC']);axs[0].axhline(1,ls='--',color='k',lw=.8);axs[0].set(xlabel='$q$',ylabel=r'$K(q) \log(1/q)/\pi$',title='Critical nonanalyticity')
    rr=data(0,'critical_crossover');ss=sorted(set(r['s'] for r in rr));axs[1].semilogx(ss,[next(r['theory'] for r in rr if r['s']==s) for s in ss],'k-',label='limit')
    for m,c in [(1e-4,'#d49932'),(1e-8,COL['SARC'])]:
        for side,ls in [(-1,'--'),(1,'-')]:
            a=sorted([r for r in rr if r['m']==m and r['side']==side],key=lambda r:r['s']);axs[1].semilogx([r['s'] for r in a],[r['exact_scaled'] for r in a],ls,color=c,label=f'$m={m:g}$' if side==1 else None)
    axs[1].set(xlabel='$q/m$',ylabel='scaled kernel difference',title='Nonpolynomial crossover');axs[1].legend(fontsize=7)
    rr=data(0,'bounded_range_scaling')
    for R,c in [(1,COL['LDA']),(8,COL['SARC-R8'])]:
        a=sorted([r for r in rr if r['R']==R and r['side']==1],key=lambda r:r['mass']);axs[2].loglog([1/r['mass'] for r in a],[r['scaled_prediction_sup'] for r in a],'o-',color=c,label=f'R={R}')
    axs[2].set(xlabel='$1/m$',ylabel='scaled finite-range norm',title='Fixed bounded range');axs[2].legend(fontsize=8);finish(fig,'01_no_go')
    fig,axs=plt.subplots(1,3,figsize=(10.4,3.1),layout='constrained');orders=data(2,'weak_field_orders')
    for ax,metric,label in zip(axs,['density_rms','euler_defect_rms','relaxation_penalty'],['Density error','Euler defect','Same-functional relaxation']):
        for i,m in enumerate(COL):
            vals=[r['slope'] for r in orders if r['method']==m and r['metric']==metric and r['window']=='primary'];ax.scatter(i+np.linspace(-.14,.14,len(vals)),vals,s=16,color=COL[m]);ax.plot([i-.2,i+.2],[np.median(vals)]*2,'k-',lw=.8)
        ax.set(xticks=range(3),xticklabels=list(COL),ylabel='error exponent',title=label)
    finish(fig,'02_cancellation')
    rows=data(3,'window_scan');fixed=[r for r in rows if r['protocol']=='fixed_period'];ms=sorted(set(r['m'] for r in fixed));fig,axs=plt.subplots(1,2,figsize=(10.4,4),layout='constrained')
    for ax,kind,xkey,label in [(axs[0],'scaled','alpha',r'$\alpha=\lambda\|v\|_\infty/m$'),(axs[1],'absolute','lambda',r'fixed $\lambda$')]:
        aa=sorted(set(r[xkey] for r in fixed if r['amplitude_kind']==kind));z=np.array([[max(r['relative_error'] for r in fixed if r['m']==m and r['amplitude_kind']==kind and r[xkey]==a) for a in aa] for m in ms]);pc=ax.pcolormesh(edge(aa),edge(ms),z,norm=LogNorm(1e-5,.5),cmap='viridis',shading='flat');cs=ax.contour(aa,ms,z,levels=[.01,.05],colors=['white','#f4b15b'],linewidths=1.2);ax.clabel(cs,fmt={.01:'1%',.05:'5%'},fontsize=8)
        for m in ms:
            for a in aa:
                if any(not r['solver_valid'] for r in fixed if r['m']==m and r['amplitude_kind']==kind and r[xkey]==a):ax.plot(a,m,'rx',ms=8)
        ax.set(xscale='log',yscale='log',xlabel=label,ylabel='$m=|g_0-1|$',title='Fixed period L=32; worst of four cases')
    fig.colorbar(pc,ax=axs,shrink=.8,label='error / induced density response');finish(fig,'03_applicability_map')
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.7),layout='constrained')
    for lam in [.0001,.001,.003,.01,.03]:
        axs[0].loglog(ms,[max(r['relative_error'] for r in fixed if r['m']==m and r['amplitude_kind']=='absolute' and r['lambda']==lam) for m in ms],'o-',label=rf'$\lambda={lam:g}$',ms=4)
    for a,c in [(.1,'#087e91'),(.3,'#d49932'),(1,'#a34f79')]:
        for protocol,ls in [('fixed_period','--'),('qxi_one','-')]:
            rr=[r for r in rows if r['protocol']==protocol and r['amplitude_kind']=='scaled' and r['alpha']==a];mm=sorted(set(r['m'] for r in rr));axs[1].loglog(mm,[max(r['relative_error'] for r in rr if r['m']==m) for m in mm],ls+'o',color=c,label=f'a={a:g}, '+('fixed L' if protocol=='fixed_period' else 'q/m=1'),ms=4)
    for ax in axs:
        ax.axhline(.01,color='k',ls=':',lw=1);ax.axhline(.05,color='#8d7435',ls=':',lw=1);ax.set(xlabel='$m$',ylabel='worst relative density error');ax.legend(fontsize=7)
    axs[0].set_title('Fixed amplitude: critical approach');axs[1].set_title('Different wavelength protocols');finish(fig,'04_critical_paths')
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.7),layout='constrained');env=data(3,'conservative_windows')
    for protocol,ls in [('fixed_period','--'),('qxi_one','-')]:
        for t,c in [(.01,'#087e91'),(.05,'#a34f79')]:
            a=sorted([r for r in env if r['protocol']==protocol and r['tolerance']==t],key=lambda r:r['m']);label=f'{100*t:g}%, '+('fixed L' if protocol=='fixed_period' else 'q/m=1')
            for ax,key in [(axs[0],'lambda_verified_all'),(axs[1],'alpha_verified_all')]:ax.loglog([r['m'] for r in a],[r[key] for r in a],ls+'o',color=c,label=label,ms=4)
    axs[0].set(xlabel='$m$',ylabel=r'verified $\lambda_{max}$',title='Conservative sampled window');axs[1].set(xlabel='$m$',ylabel=r'verified $\alpha_{max}$',title='Peak-amplitude / mass window')
    for ax in axs:ax.legend(fontsize=8)
    finish(fig,'05_verified_windows')
    weak=data(3,'weak_nonlinear_coefficients');fig,axs=plt.subplots(1,2,figsize=(10.4,3.7),layout='constrained')
    for side,c in [(-1,'#087e91'),(1,'#a34f79')]:
        for family,ls in [('four_low_modes','-'),('localized_packet','--')]:
            rr=sorted([r for r in weak if r['protocol']=='fixed_period' and r['side']==side and r['family']==family and r.get('C2')],key=lambda r:r['m']);label=('g<1' if side<0 else 'g>1')+', '+family.replace('_',' ')
            axs[0].loglog([r['m'] for r in rr],[r['C2'] for r in rr],ls+'o',color=c,ms=4,label=label);axs[1].semilogx([r['m'] for r in rr],[r['slope'] for r in rr],ls+'o',color=c,ms=4)
    axs[0].set(xlabel='$m$',ylabel=r'$C_2\simeq\|\delta\|/\lambda^2$',title='Remaining nonlinear coefficient');axs[0].legend(fontsize=7);axs[1].axhline(2,color='k',ls=':',lw=.8);axs[1].set(xlabel='$m$',ylabel='resolved weak-field exponent',title='Second order does not bound the coefficient');finish(fig,'06_nonlinear_coefficient')

def window_table(env,protocol):
    rows=[]
    for m in sorted(set(r['m'] for r in env if r['protocol']==protocol),reverse=True):
        a=next(r for r in env if r['protocol']==protocol and r['m']==m and r['tolerance']==.01);b=next(r for r in env if r['protocol']==protocol and r['m']==m and r['tolerance']==.05);rows.append([f'{m:.5g}',f'{a["lambda_verified_all"]:.3g}',f'{b["lambda_verified_all"]:.3g}',f'{a["alpha_verified_all"]:.3g}',f'{b["alpha_verified_all"]:.3g}'])
    return rows

def documents():
    env=data(3,'conservative_windows');summary=json.loads((NEW/'data/summary.json').read_text())
    texts=[
'''# 1 / No-Go Theorems

| 实验 | 目的 | 简洁结论 |
| --- | --- | --- |
| N1：临界核 | 检验有限范围 Hessian 的解析性障碍 | 临界 K≈π/log(1/|q|) 无法由固定有限三角多项式精确表示 |
| N2：交叉标度 | 检验 q=ms 联合极限的非多项式结构 | 有界固定范围缩放趋零；允许奇异系数也只能得到固定有限维多项式极限 |

保留临界核、两侧交叉数据、有界范围对照和最近邻两点常数。未复制高阶 minimax 拟合及系数竞赛。结论约束精确结构，不排除有限窗口高精度近似或范围随临界长度增长。
''',
'''# 2 / Minimal Nonlocal Correction

| 验证 | 目的 | 简洁结论 |
| --- | --- | --- |
| M1：核差构造 | 固定精确均匀 LDA 后识别唯一缺失的二阶信息 | ΔK=Kexact(q;r)−Kexact(0;r) 由完整 Hessian 要求唯一确定 |
| M2：变分与对称性 | 核对完整 RMS 导数、均匀 Hessian 与局域符号不变性 | SARC 是变分一致的一项非局域修复 |
| M3：有限范围重新拟合 | 排除固定范围模型只是系数未调好的解释 | 重拟合降低核误差，但 R=8 的密度误差仍为一阶；SARC 为二阶 |

主泛函：F=Σ fhom(ρi)+(8Nr²)⁻¹Σq≠0 ΔK(q;r)|uq|²，u=ρ²，r²=〈ρ²〉。不新增拟合系数。平方坐标适配对称性，RMS 是选定的延拓约定；唯一性只到二阶核，不到完整非线性泛函。

`data/structural_validation.csv` 保留直接相关的 10 项既有检查；`data/structural_checks.csv` 保存 12 项广域核与热力学求解器检查。

重新拟合对照保留 R=1/2/4/8 的全部 24 组核结果；R=8 另做 24 组波形、四个幅度，共 96 条弱场验证。系数按最大相对核误差拟合，不加系数约束；验证动量与拟合动量分开。密度误差指数为 0.9933–1.0074，仍有领先线性项。SARC 用一项完整核差消去该项，体现最小必要非局域修复的作用。

第五轮原始重拟合文件已在此前清理中删除；当前对照是根据保留精确核与 JW 密度重建的实验。设置见 `config/finite_range_refit.json`，标量结果见 `data/finite_range_refit_*.csv`，96 条密度剖面见 `profiles/finite_range_refit_profiles.json`，来源哈希与运行记录见 `logs/finite_range_refit.json`。
''',
'''# 3 / Error-Cancellation Mechanism

| 实验 | 目的 | 简洁结论 |
| --- | --- | --- |
| E1：24 组多模弱场 | 用独立非均匀 JW 解验证消除线性项 | SARC 密度误差指数 1.99861–2.00190；Euler 缺陷指数 1.99898–2.00093 |
| E2：尾部消融 | 区分核结构修复与一般精度提升 | R8 截断恢复一阶误差；其系数由均匀核准确预测 |

只保留 LDA、SARC-R8、SARC，共 288 条结果与完整正负场剖面；删除重复 GGA 对照。保留 54 行既有尾部因果证据，包括未分辨标记。已有单模、双模重复验证不再单独复制。

在 TFIM 根目录运行 `python -m tfim.weak_field` 可独立重算全部 288 条弱场结果，并与保留数据核对。配置在 `config/experiment.json`，必要核输入在 `cache/`，完整密度在 `profiles/weak_field_profiles.json`。临时重算输出在核验后清理，只保留 `logs/reproduction_check.json`；不依赖已删除的旧项目。

同一近似泛函内部的松弛差为四阶，仅作为误差机制的补充；不能称真实基态能量达到四阶精度。R8 仍含全局 RMS，只是均匀 Hessian 有限范围。
''',
'''# 4 / Critical Applicability and Limitations

| 实验 | 目的 | 简洁结论 |
| --- | --- | --- |
| A1：联合窗口扫描 | 量化残余非线性误差何时可忽略 | 使用诱导密度响应归一化的 1%/5% 容差，给出逐组通过—失败边界 |
| A2：两条临界路径 | 区分固定波长与 q/m=1，区分固定 λ 与固定 λ/m | 固定幅度时误差可随临界逼近显著放大；不存在由二阶阶数单独保证的窗口 |
| A3：弱场余项及数值边界控制 | 解释 C₂ 的质量依赖，排除积分、核表和起点伪影 | 二阶误差与不断增大的系数可同时存在，窗口须直接测量 |
| A4：保留高阶响应与分支反例 | 防止把局部修复误读成精确/全局泛函 | 二阶谐波仍有偏差，多稳定分支仍可能存在 |

ερ=RMS(ρSARC−ρexact)/RMS(ρexact−ρhom)，α=λ max|v|/m，v 零均值且 RMS=1。窗口是已测网格上从小幅度开始的连续通过前缀，不是任意波形或连续域保证。严格临界 m=0 不在本实验内。

固定 L=32、λ=0.001：最大相对误差在 m=0.003 为 0.656%，m=0.001 为 1.288%，m=0.0003 为 2.588%，m=0.0001 为 6.479%。因此 1% 标准下需要比 m=0.001 更远离临界，或同步降低扰动；实际边界只由相邻已测点夹定。

## 保守窗口：固定 L=32

| m | 1%：λ | 5%：λ | 1%：α | 5%：α |
| --- | --- | --- | --- | --- |
''']
    texts[3]+='\n'.join('| '+' | '.join(r)+' |' for r in window_table(env,'fixed_period'))+'\n\n## 保守窗口：q/m=1\n\n| m | 1%：λ | 5%：λ | 1%：α | 5%：α |\n| --- | --- | --- | --- | --- |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in window_table(env,'qxi_one'))+f'\n\n主扫描 620 行，阈值边界另加密 {summary["boundary_refinement_rows"]} 行，总计 {summary["rows"]} 行。几何二分使通过—失败幅度比不超过 1.12；不改变泛函或容差。详细边界、右删失标记与不确定点见 `data/applicability_windows.csv`；完整结果见 `data/window_scan.csv`；全部密度见 `profiles/`。两侧与两波形的包络仅适用于本采样集合。\n\n作为保守起点，峰值扰动 ≤0.2m 在本实验的 200 个相应采样点中，最坏相对误差为 0.862%，均低于 1%；这不是对任意波形的保证。不同周期使用相同的无量纲四模权重/相位和波包形状，以隔离尺度效应。\n'
    from .workspace import documentation,directory_notes
    for i,(p,text) in enumerate(zip(PARTS,texts),1):(ROOT/p/'README.md').write_text(text+documentation(i),encoding='utf8')
    directory_notes()
    root='''# TFIM：最小必要非局域修复

从两条 No-Go 出发，以一个 SARC 非局域修正补回缺失均匀逆响应核；验证误差消除，并量化临界逼近时残余非线性误差的可用窗口。

[完整 PDF 报告（含重拟合对照）](report/TFIM_核心研究与临界适用窗口_含重拟合对照.pdf) · [可编辑正文](report/TFIM_report.md) · [目录与运行入口](DIRECTORY_STRUCTURE.md) · [来源清单](provenance.json) · [剔除清单](selection.json)

## 1 / No-Go Theorems

[必要性：临界非解析核与交叉函数](01_No-Go_Theorems/README.md)。只保留支持两条定理的证据。

## 2 / Minimal Nonlocal Correction

[最小二阶核差、完整变分与有限范围重拟合对照](02_Minimal_Nonlocal_Correction/README.md)。重拟合有限范围系数可降低误差，但仍保留一阶密度误差；SARC 用一项完整核差消除该项。

## 3 / Error-Cancellation Mechanism

[24 组弱场与纯尾部消融](03_Error-Cancellation_Mechanism/README.md)。保留 288 条关键结果及独立复现代码，去除重复对照和候选模型实验。

## 4 / Critical Applicability and Limitations

[新增 620 个外场的适用窗口](04_Critical_Applicability_and_Limitations/README.md)。以误差占诱导密度响应的 1%/5% 定义“可以忽略”，同时检验临界距离、扰动幅度和波长。

固定 L=32、λ=0.001，最坏相对误差由 m=0.1 时约 0.0855% 增至 m=0.0001 时约 6.479%。**消除领先线性误差，不保证固定幅度下无限逼近临界点仍然准确。**

## 复现与目录边界

在本目录运行：

```powershell
python run.py
```

默认依次执行结构测试、新临界扫描、窗口提取、边界数值控制、报告和审计。也可使用 `python run.py --stages analyze report audit`。第三部分全部 288 条保留弱场结果可用 `python -m tfim.weak_field` 独立重算；第四部分可用默认命令重算。原始密度、关键反例记录和必要核输入均已迁入 TFIM，不依赖任何旧目录。

四个子项目均使用 `config / data / profiles / cache / figures / logs`，各有 `README.md` 和 `run.py`。进入任一子项目运行 `python run.py` 默认核验已有数据；可重算阶段见 [目录与运行入口](DIRECTORY_STRUCTURE.md)。第三、四部分的实验配置分别位于各自的 `config/experiment.json`，跨项目任务日志在根目录 `logs/`。

`provenance.json` 给出保留证据与来源哈希；`audit.json` 检查数据、公式对应的观测量、窗口分类和报告。PDF 需要 Typst 及 SimSun/Cambria/Cambria Math，审计需要 PyMuPDF；Python 依赖见 `requirements.txt`。

TFIM 是唯一研究项目。原 `tfim_critical_dft` 目录与 PROMPT1–5 探索文档已删除。保留有限范围重新拟合，作为 SARC 最小必要性的对照；该实验由现存核表重建，未恢复第五轮原文件。PLR/H2/L2、与结构验证无关的高阶拟合和重复排名不保留。`provenance.json` 中的历史路径用于溯源；重拟合的输入哈希另记在第二部分运行日志中。
'''
    root=root.replace('新增 620 个外场的适用窗口',f'新增 620 个主网格外场与 {summary["boundary_refinement_rows"]} 个边界加密点的适用窗口').replace('窗口提取、边界数值控制','窗口提取、阈值边界加密、边界数值控制')
    (ROOT/'README.md').write_text(root,encoding='utf8')
    (ROOT/'requirements.txt').write_text('numpy==2.2.6\nscipy==1.15.2\nmatplotlib==3.9.1\nthreadpoolctl>=3\nPyMuPDF==1.25.2\n',encoding='utf8')
def run():
    figures()
    from .report_layout import build
    build()
    documents()
    print('Four-part report and focused directory documentation generated',flush=True)
