"""Balanced report: 2/3/4/4 scientific pages with traceable, real data tables."""
import csv,json,re,subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import fitz
from .common import ROOT,NEW,PARTS,read,save,sha

REPORT=ROOT/'report'
SECTION={1:'1 / No-Go Theorems',2:'2 / Minimal Nonlocal Correction',3:'3 / Error-Cancellation Mechanism',4:'4 / Critical Applicability and Limitations'}
FAMILY={'four_low_modes':'F','mixed_scales':'M','localized_packet':'P','twelve_modes':'T'}
COL={'LDA':'#7a818b','GGA':'#c28224','NRC-LDA-R8':'#a34f79','NRC-LDA':'#087e91'}
def esc(x):
    s=str(x)
    for ch in ['\\','[',']','#','$','<','>','*','_','@']:s=s.replace(ch,'\\'+ch)
    return s
def num(x,style='s'):
    if x is None:return '—'
    if isinstance(x,str):return x
    if isinstance(x,(bool,np.bool_)):return '是' if x else '否'
    if style=='i':return str(int(x))
    if style=='f':return f'{float(x):.6f}'
    if style=='p':return f'{100*float(x):.5f}'
    if style=='rho':return f'{float(x):.10f}'
    if style=='g':return f'{float(x):.6g}'
    return f'{float(x):.5e}'

class Evidence:
    def __init__(self):self.sources={};self.tables=[]
    def source(self,path):
        rel=Path(path).relative_to(ROOT).as_posix()
        if rel not in self.sources:self.sources[rel]=dict(id=f'S{len(self.sources)+1:02}',path=rel,sha256=sha(path))
        return self.sources[rel]['id']
    def csv(self,part,name):
        path=ROOT/PARTS[part-1]/'data'/f'{name}.csv';sid=self.source(path)
        return [dict(r,_ref=dict(source=sid,kind='csv',row=i+2,snapshot=r)) for i,r in enumerate(read(path))]
    def json(self,path):
        path=Path(path);sid=self.source(path);return json.loads(path.read_text(encoding='utf8')),sid
    def npz(self,path):return np.load(path),self.source(path)

class Document:
    def __init__(self,book):self.book=book;self.md=[];self.typ=[];self.pages=[]
    def page(self,section,title):
        if self.typ:self.typ.append('#pagebreak()')
        self.pages.append(dict(section=section,title=title))
        if section:self.typ.append('#text(size: 9pt,fill: rgb("607785"))['+SECTION[section]+']\n#v(2mm)')
        self.md.append('\n## '+title+'\n');self.typ.append('= '+esc(title)+'\n')
    def p(self,text):self.md.append(text+'\n');self.typ.append(esc(text)+'\n')
    def note(self,text):self.md.append(text+'\n');self.typ.append('#block(width: 100%,inset: (x: 7pt,y: 5pt),fill: rgb("f1f5f6"))[#text(size: 9pt)['+esc(text)+']]\n')
    def eq(self,md,typ):self.md.append('\\[\n'+md+'\n\\]\n');self.typ.append('$ '+typ+' $\n')
    def table(self,tid,title,headers,keys,records,styles,rule,note='',size=8.7):
        values=[[r[k] for k in keys] for r in records];printed=[[num(v,st) for v,st in zip(row,styles)] for row in values]
        self.md.append(f'**表 {tid}　{title}**\n');self.md+=['| '+' | '.join(h.replace('|',r'\|') for h in headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(c.replace('|',r'\|') for c in r)+' |' for r in printed]+['']
        self.typ.append('#block(above: 5pt,below: 4pt)[#text(weight: "bold",size: 10pt)['+esc('表 '+tid+'　'+title)+']]')
        cells=',\n'.join('['+esc(c)+']' for row in printed for c in row)
        head=', '.join('['+esc(c)+']' for c in headers)
        widths=['.5fr','.6fr','3fr','1.4fr'] if tid=='D.1' else ['1fr']*len(headers)
        self.typ.append('#block(width: 100%)[#set text(size: '+str(max(size,9))+'pt)\n#table(columns: ('+', '.join(widths)+'), inset: (x: 3pt,y: 3.5pt), stroke: .35pt + rgb("cfdae0"), fill: (x,y) => if y == 0 {rgb("e8f0f3")} else if calc.rem(y,2) == 0 {rgb("f6f8fa")}, table.header('+head+'),\n'+cells+')]\n')
        source_ids=sorted({ref['source'] for r in records for ref in r.get('_refs',[])})
        caption=('来源 '+', '.join(source_ids)+'；' if source_ids else '')+rule+(' '+note if note else '')
        self.md.append(caption+'\n');self.typ.append('#text(size: 8.4pt,fill: rgb("52626b"))['+esc(caption)+']\n')
        directory=REPORT/'data_tables';directory.mkdir(exist_ok=True)
        with (directory/f'table_{tid}.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.writer(f);w.writerow(keys);w.writerows(values)
        self.book.tables.append(dict(id=tid,title=title,section=self.pages[-1]['section'],headers=headers,keys=keys,styles=styles,records=records,printed=printed,selection_rule=rule,note=note,export=f'data_tables/table_{tid}.csv'))
    def fig(self,part,name,caption,height=None):
        rel='../'+PARTS[part-1]+'/figures/'+name+'.png';self.md.append(f'![{caption}]({rel})\n')
        self.typ.append('#figure(image("'+rel+'",width: 100%'+(f',height: {height}mm,fit: "contain"' if height else '')+'),caption: [#text(size: 9pt)['+esc(caption)+']])\n')
    def save(self):
        pre='''#set document(title: "TFIM：最小非局域修复与临界适用窗口",author: "TFIM computational study")
#set page(paper: "a4",margin: (x: 18mm,top: 16mm,bottom: 17mm),numbering: "1",footer: context [#text(size: 9pt,fill: rgb("637580"))[#align(center)[TFIM · #counter(page).display() / #counter(page).final().first()]]])
#set text(font: ("SimSun","Cambria"),size: 10.5pt,lang: "zh")
#set par(justify: true,leading: .55em,spacing: 7pt)
#show heading.where(level: 1): set text(size: 16pt,fill: rgb("173e55"))
#show math.equation: set text(font: "Cambria Math")
#set figure(gap: 5pt)
'''
        (REPORT/'TFIM_report.md').write_text('\n'.join(self.md),encoding='utf8');(REPORT/'TFIM_report.typ').write_text(pre+'\n'.join(self.typ),encoding='utf8')
        save(REPORT/'table_provenance.json',dict(sources=list(self.book.sources.values()),tables=self.book.tables,planned_pages=self.pages))
        subprocess.run(['typst','compile','--root',str(ROOT),str(REPORT/'TFIM_report.typ'),str(REPORT/'TFIM_核心研究与临界适用窗口_含重拟合对照.pdf')],check=True)

def ref(r):return [r['_ref']]
def rec(r,**values):return dict(values,_refs=ref(r))
def record_no(r):return r['_ref']['row']

def extra_figures(weak,tail):
    folder=ROOT/PARTS[2]/'figures'
    fig,axs=plt.subplots(1,3,figsize=(10.4,2.75),layout='constrained')
    for ax,key,label in zip(axs,['density_rms','euler_defect_rms','relaxation_penalty'],['Density error','Euler defect','Relaxation penalty']):
        for method,c in COL.items():
            rr=sorted([r for r in weak if r['L']==128 and r['side']==1 and r['family']=='mixed_scales' and r['method']==method],key=lambda r:r['ratio']);ax.loglog([r['ratio'] for r in rr],[r[key] for r in rr],'o-',color=c,ms=4,label=method)
        ax.axvspan(.00028,.0032,color='#eaf4f4',zorder=-5);ax.set(xlabel=r'$\lambda/m$',ylabel=label,title=label);ax.legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(folder/f'07_recorded_error_curves.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10.4,2.8),layout='constrained')
    for method in ['LDA','NRC-LDA-R8']:
        rr=[r for r in weak if r['method']==method and r['ratio']==.0003];axs[0].loglog([r['predicted_linear_density_coefficient'] for r in rr],[r['measured_odd_density_coefficient'] for r in rr],'o',color=COL[method],ms=3,label=method)
    axs[0].plot([.0001,1],[.0001,1],'k--',lw=.8);axs[0].set(xlabel='predicted linear coefficient',ylabel='measured odd error / amplitude',title='48 independent nonzero-coefficient tests');axs[0].legend(fontsize=7)
    for side,c in [(-1,'#087e91'),(1,'#a34f79')]:
        rr=sorted([r for r in tail if r['L']==128 and r['side']==side and r['R']>0],key=lambda r:r['R']);axs[1].loglog([r['R_over_xi'] for r in rr],[max(r['predicted_linear_RMS'],1e-16) for r in rr],color=c,label=f'side {side:+d}')
        ss=[r for r in rr if r['linear_signal_resolved']];axs[1].loglog([r['R_over_xi'] for r in ss],[r['measured_odd_over_lambda'] for r in ss],'o',ms=3,color=c)
    axs[1].set(xlabel=r'$R/\xi$',ylabel='linear density coefficient',title='Pure tail removal: L=128');axs[1].legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(folder/f'08_recorded_tail_causality.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)

def build():
    scientific=[p for part in PARTS for p in (ROOT/part).rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.jsonl','.npz'] and 'logs' not in p.parts]
    before={str(p.relative_to(ROOT)):sha(p) for p in scientific}
    b=Evidence();critical=b.csv(1,'critical_kernel');cross=b.csv(1,'critical_crossover');bounded=b.csv(1,'bounded_range_scaling');validation=b.csv(2,'structural_validation');newtests=b.csv(2,'structural_checks');weak=b.csv(3,'weak_field_results');orders=b.csv(3,'weak_field_orders');tail=b.csv(3,'tail_causality');scan=b.csv(4,'window_scan');windows=b.csv(4,'applicability_windows');env=b.csv(4,'conservative_windows');bc=b.csv(4,'boundary_controls');coeff=b.csv(4,'weak_nonlinear_coefficients');nonlinear=b.csv(4,'retained_nonlinear_limit');branches=b.csv(4,'retained_branch_endpoints')
    weak+=b.csv(3,'gga_weak_field');orders+=b.csv(3,'gga_orders');gga_critical=b.csv(4,'gga_critical_path')
    gga_summary=json.loads((ROOT/PARTS[2]/'data/gga_summary.json').read_text())
    summary=json.loads((NEW/'data/summary.json').read_text());control=json.loads((NEW/'data/control_summary.json').read_text())
    extra_figures(weak,tail);d=Document(b)

    d.page(0,'TFIM：最小必要非局域修复')
    d.p('四部分研究报告 · 数据记录版 · 2026 年 11 月 17 日')
    d.p('本研究寻找修复临界误差所需的最小非局域项。两条 No-Go 定理说明局域和半局域泛函缺少什么；据此构造 NRC-LDA，再用精确解检验误差如何降低，以及修复在多近的临界距离内有效。')
    d.p('NRC-LDA 的全称为 Nonlocal Response-Corrected LDA。它在均匀 LDA 上增加一项核差修正，补回缺失的均匀逆响应，不引入新的拟合系数：')
    d.eq(r'F_{\rm NRC-LDA}=\sum_i f_{\rm hom}(\rho_i)+\frac{1}{8Nr^2}\sum_{q\ne0}\Delta K(q;r)|u_q|^2,\quad u_i=\rho_i^2,\quad r^2=\langle\rho^2\rangle.','F_("NRC-LDA")=sum_i f_("hom")(rho_i)+1/(8N r^2) sum_(q != 0) Delta K(q;r)|u_q|^2,quad u_i=rho_i^2,quad r^2=angle.l rho^2 angle.r.')
    d.note('主要结果：NRC-LDA 将弱场密度误差从一阶降为二阶。接近临界点时，剩余二阶误差的系数仍会增大，因此需要同时控制临界距离和扰动幅度。第四部分给出 1% 和 5% 误差标准下的适用窗口。')
    d.p('机制验证包含 288 条结果，窗口实验包含 837 条结果。另加入有限范围模型重新拟合的 24 组核对照和 96 条弱场结果，检验调节系数能否替代完整的非局域修复。')
    d.p('文献 GGA 对照采用 Mao 等（2021）的原式和固定参数，新增 96 条弱场结果、28 条临界扫描，用于区分误差减小与领先误差消除。')
    d.p('报告结构')
    d.p('1 / No-Go Theorems（第 2–3 页）：有限范围修正为何无法恢复临界响应。')
    d.p('2 / Minimal Nonlocal Correction（第 4–6 页）：构造 NRC-LDA，检验变分实现，并与重新拟合的有限范围模型比较。')
    d.p('3 / Error-Cancellation Mechanism（第 7–10 页）：线性误差如何消除，非局域尾部起什么作用。')
    d.p('4 / Critical Applicability and Limitations（第 11–14 页）：剩余误差何时可忽略，修复还有哪些不足。第 15 页为数据索引。')
    d.note('读表约定：F=四低频模，M=混合尺度，P=局域波包，T=十二模；“侧”指 g₀=1±m；“—”表示无数据或无可靠拟合。“行”是原 CSV 行号，表头为第 1 行。完整精度数据见配套 CSV。')

    d.page(1,'1.1　有限范围修正为何在临界处失效')
    d.p('考虑周期 TFIM，取 J=1。ρ 表示横向密度，χ 表示均匀响应，K=χ⁻¹ 是泛函的二阶导数。第一条 No-Go 比较精确临界核与有限范围修正的小动量行为。')
    d.eq(r'H=-\sum_i\sigma_i^z\sigma_{i+1}^z-\sum_i g_i\sigma_i^x,\quad E[\rho;g]=F[\rho]-\sum_i g_i\rho_i.','H=-sum_i sigma_i^z sigma_(i+1)^z-sum_i g_i sigma_i^x,quad E[rho;g]=F[rho]-sum_i g_i rho_i.')
    d.eq(r'K(q;1)\sim\frac\pi{\log(1/|q|)},\qquad\Delta K_R(q)=2\sum_{\ell=1}^R b_\ell[1-\cos(\ell q)].','K(q;1) tilde pi/log(1/|q|),quad Delta K_R(q)=2 sum_(ell=1)^R b_ell[1-cos(ell q)].')
    d.p('在平移、反射对称下，固定范围 R、有限系数的修正是偶三角多项式，零动量附近按 q 的偶次幂展开。精确临界核却按 1/log(1/|q|) 变化。两者的函数结构不同，因此固定有限范围修正无法恢复 q→0 的临界响应。')
    rr=sorted(critical,key=lambda r:r['q']);chosen=[rr[i] for i in np.linspace(0,len(rr)-1,8).round().astype(int)]
    records=[rec(r,line=record_no(r),q=r['q'],K=r['K_exact'],Klda=r['K_LDA'],ratio=r['K_exact']*np.log(1/r['q'])/np.pi) for r in chosen]
    d.table('1.1','临界核与对数归一化',['行','q','Kexact','KLDA','K log(1/q)/π'],['line','q','K','Klda','ratio'],records,['i','s','s','s','f'],'按 q 排序后等索引抽取 8 行；末列由原始 q、K 计算。')
    d.fig(1,'01_no_go','左：小 q 下的对数渐近；中：临界点两侧的交叉函数；右：有界有限范围修正的缩放值趋零。',height=57)
    d.note('临界核需要有限范围展开所缺少的非解析结构。表 1.1 中，q 越小，对数归一化值越接近 1。')

    d.page(1,'1.2　临界逼近中的交叉函数')
    d.p('第二条 No-Go 考察临界距离与波长同时变化的情形。令 m=|g−1|、q=ms，在固定区间 0<a≤s≤b 上，精确缩放核趋于同一个非多项式函数：')
    d.eq(r'\log^2(1/m)\Delta K(ms;m)\to\mathcal S(s)=\frac\pi2\int_0^1\log[1+x(1-x)s^2]dx.','log^2(1/m) Delta K(m s;m) arrow S(s)=pi/2 integral_0^1 log[1+x(1-x)s^2] dif x.')
    s0=min({r['s'] for r in cross},key=lambda s:abs(np.log(s)));rr=sorted([r for r in cross if r['s']==s0],key=lambda r:(-r['m'],r['side']))
    d.table('1.2','同一 s 下的双侧缩放结果',['行','m','侧','精确缩放值','极限 S','差值'],['line','m','side','exact','limit','residual'],[rec(r,line=record_no(r),m=r['m'],side=r['side'],exact=r['exact_scaled'],limit=r['theory'],residual=r['exact_residual']) for r in rr],['i','s','i','f','f','s'],f'取最接近 1 的网格点 s={s0:.10f}，列出两侧、四个 m 的结果。')
    d.p('固定 R 且 Σ|bℓ|≤B 时，由 |1−cos x|≤x²/2 得到 |ΔKᵣ(ms)|≤BR²b²m²。乘上 log²(1/m) 后仍趋零，与上面的非零极限不同。表 1.3 展示两个范围、临界点两侧的结果。')
    rr=sorted(bounded,key=lambda r:(r['side'],r['R'],-r['mass']))
    d.table('1.3','有界有限范围修正',['行','m','侧','R','Σ|bℓ|','缩放预测 sup','缩放误差 sup'],['line','m','side','R','norm','prediction','error'],[rec(r,line=record_no(r),m=r['mass'],side=r['side'],R=r['R'],norm=r['coefficient_l1'],prediction=r['scaled_prediction_sup'],error=r['scaled_error_sup']) for r in rr],['i','s','i','i','f','s','f'],'两侧、两个范围，共 16 行；sup 表示所测 s 区间上的最大值。',size=8.5)
    d.p('即使允许系数在 m→0 时发散，固定 R 的限制仍然存在。令 xₘ=4sin²(ms/2)/m²，缩放核可写成 xₘ 的 R 阶多项式；若它有有限极限，该极限只能是 s²,…,s²ᴿ 的线性组合，无法等于非多项式 S(s)。')
    bound,sid=b.json(ROOT/PARTS[0]/'data/two_point_bound.json')
    d.note(f'例如 R=1 时，仅比较 s=1、2 两点，任意 c s² 的最大误差就至少为 |4S(1)−S(2)|/5={float(bound["two_point_lower_bound"]):.10f}。来源 {sid}。')

    d.page(2,'2.1　用一项核差补回缺失响应')
    d.p('两条 No-Go 指向同一个缺口：有限范围修正无法补齐临界二阶响应。均匀 LDA 已给出 K₀=f″hom=Kexact(0)，因此所需的二阶修正就是精确核与零动量核之差：')
    d.eq(r'\Delta K(q;r)=K_{\rm exact}(q;r)-K_{\rm exact}(0;r),\quad\delta^2G[r](\eta,\eta)=N^{-1}\sum_q\Delta K(q;r)|\eta_q|^2.','Delta K(q;r)=K_("exact")(q;r)-K_("exact")(0;r),quad delta^2 G[r](eta,eta)=1/N sum_q Delta K(q;r)|eta_q|^2.')
    d.p('局域 z 旋转可以翻转横向密度的符号而不改变 ZZ 相互作用。取 u=ρ² 保持这一对称性，并以 r=√〈ρ²〉作为背景。均匀态附近 δu=2rδρ，因而下式恰好补入所需的二次能量；均匀密度下，修正项为零。')
    d.eq(r'F=\sum_i f_{\rm hom}(\rho_i)+\frac1{8Nr^2}\sum_{q\ne0}\Delta K(q;r)|u_q|^2.','F=sum_i f_("hom")(rho_i)+1/(8N r^2) sum_(q != 0) Delta K(q;r)|u_q|^2.')
    kernel_records=[]
    for side in [-1,1]:
        z,sid=b.npz(ROOT/PARTS[2]/'cache'/f'K_L128_side{side}_fft128_nodes25.npz')
        for k in [0,1,2,16,32,64]:
            kernel_records.append(dict(side=side,g=float(z['fields'][12]),rho=float(z['rho'][12]),k=k,K0=float(z['K'][12,0]),K=float(z['K'][12,k]),delta=float(z['K'][12,k]-z['K'][12,0]),_refs=[dict(source=sid,kind='npz',index=[12,k],arrays=['fields','rho','K'])]))
    d.table('2.1','L=128 的均匀核输入',['侧','g₀','ρ₀','k','K(0)','K(qk)','ΔK'],['side','g','rho','k','K0','K','delta'],kernel_records,['i','f','rho','i','f','f','s'],'两侧核表的中心背景节点；固定 k=0,1,2,16,32,64，qk=2πk/128。','ΔK=K(qk)−K(0)。')
    d.note('NRC-LDA 用一项修正匹配完整均匀响应，无需新增拟合系数。二阶核差由响应匹配唯一确定，平方密度与 RMS 背景则给出本文采用的非线性形式。')
    d.p('文献 GGA：Mao 等，Testing density functional theory in a quantum Ising chain，Phys. Rev. B 104, 155145 (2021)，式 (13)，DOI: 10.1103/PhysRevB.104.155145。按原文固定 J_eff=0.35：')
    d.eq(r'F_{\rm GGA}=F_{\rm LDA}+0.35\sum_i(\rho_i^2-\rho_{i+1}^2)^2,\quad\Delta K_{\rm GGA}=32(0.35)r^2\sin^2(q/2).','F_("GGA")=F_("LDA")+0.35 sum_i (rho_i^2-rho_(i+1)^2)^2,quad Delta K_("GGA")=32(0.35)r^2 sin^2(q/2).')
    d.p('其均匀 Hessian 为最近邻修正，小 q 时仍按 q² 衰减，属于 No-Go 所约束的有限范围形式。以下直接比较这一 GGA 与 NRC-LDA，不重新拟合参数。')

    d.page(2,'2.2　梯度实现与数值检查')
    d.p('求解密度时，r 和核 ΔK 都随密度变化，求导需要包含这两部分。记 z=FFT(ρ²/r²−1)，w=IFFT(ΔKz)，S₀=ΣΔK|z|²、S₁=Σ∂rΔK|z|²，完整梯度为：')
    d.eq(r'\partial_iF=f_{\rm hom}^{\prime}(\rho_i)+\frac{\rho_iw_i}{2}+\frac{\rho_i}{N}\left(\frac{rS_1}{8N}-\frac{S_0}{4N}\right).','partial_i F=f_("hom")\'(rho_i)+rho_i w_i/2+rho_i/N (r S_1/(8N)-S_0/(4N)).')
    tests={'gradient':'梯度差分','Hessian':'Hessian 模态','line_integral':'梯度路径积分','local_sign_symmetry':'逐点符号对称','JW_Bloch_new_field':'稠密 JW/Bloch','thermodynamic_uniform':'均匀热力学极限','finite_ring_to_thermodynamic':'有限环/热力学','full_gradient':'完整梯度差分','uniform_Hessian':'均匀 Hessian'}
    d.table('2.2','10 项结构检查',['行','检查','侧','主要误差','能量误差'],['line','test','side','error','energy'],[rec(r,line=record_no(r),test=tests[r['test']],side=r['side'],error=r.get('error'),energy=r.get('energy_error')) for r in validation],['i','s','i','s','s'],'误差定义沿用原记录。',size=8.8)
    d.table('2.3','12 项核与求解器检查',['行','检查','m','侧','主要误差','能量误差'],['line','test','m','side','error','energy'],[rec(r,line=record_no(r),test=tests[r['test']],m=r['m'],side=r['side'],error=r['error'],energy=r.get('energy_error')) for r in newtests],['i','s','s','i','s','s'],'均匀密度与有限环比较采用密度 RMS；梯度和 Hessian 误差见对应检查。',size=8.7)
    d.note(f'梯度差分、均匀 Hessian 和独立 JW 比较均通过检查。新增 GGA 的梯度、Hessian、符号/平移对称、均匀极限与周期键计数检查共 12 项，最大误差 {gga_summary["structural_max_error"]:.3e}；完整记录见第二部分 gga_structural_checks.csv。')

    d.page(2,'2.3　重新拟合能否替代非局域修复')
    refits=b.csv(2,'finite_range_refit_kernels');refit_summary=b.csv(2,'finite_range_refit_summary')
    d.p('让有限范围模型的每个系数独立重拟合，以最小化最大相对核误差。取 R=1/2/4/8，各背景分别拟合，不加系数约束。核表的偶数动量点用于拟合，奇数点用于验证。')
    d.eq(r'K_R(q)=K_{exact}(0)+2\sum_{\ell=1}^R b_\ell[1-\cos(\ell q)],\quad\min_{\{b_\ell\}}\max_{q\in train}|K_R(q)/K_{exact}(q)-1|.', 'K_R(q)=K_("exact")(0)+2 sum_(ell=1)^R b_ell[1-cos(ell q)],quad min_(b_ell) max_(q in "train") |K_R(q)/K_("exact")(q)-1|.')
    d.table('2.4','有限范围核：直接截断与重新拟合',['L','侧','R','截断误差 (%)','拟合误差 (%)','验证误差 (%)'],['L','side','R','cut','fit','valid'],[rec(r,L=r['L'],side=r['side'],R=r['R'],cut=r['truncated_train_error'],fit=r['train_error'],valid=r['validation_error']) for r in refits],['i','i','i','p','p','p'],'m=2π/L；误差为最大相对核误差，截断与拟合使用同一组训练动量。',size=8.7)
    d.p('再将重拟合的 R=8 核代入同一泛函形式，复用第三部分的 24 组波形和四个幅度。下表给出每个背景的四组密度误差阶数范围。')
    d.table('2.5','重拟合 R=8 与 NRC-LDA 的密度误差阶数',['L','侧','预测 C₁ 最大值','测量 C₁ 最大值','重拟合 p 最小','重拟合 p 最大','NRC-LDA p 最小','NRC-LDA p 最大'],['L','side','pred','measured','pmin','pmax','smin','smax'],[rec(r,L=r['L'],side=r['side'],pred=r['predicted_C1_max'],measured=r['measured_C1_max'],pmin=r['refit_p_min'],pmax=r['refit_p_max'],smin=r['nrc_lda_p_min'],smax=r['nrc_lda_p_max']) for r in refit_summary],['i','i','s','s','f','f','f','f'],'C₁ 为一阶密度误差系数；测量取最小幅度，p 用前三个幅度拟合。',size=8.4)
    d.note('重新拟合降低了核误差，但密度误差仍为一阶（p=0.9933–1.0074）；NRC-LDA 为二阶。NRC-LDA 的最小必要性体现在用一项完整核差消去领先误差，无需增加拟合系数。')
    d.p('数据来源：第五轮原始重拟合文件已在清理中删除，本节使用保留的精确核与 JW 密度重新计算。')

    d.page(3,'3.1　弱场下的误差变化')
    d.p('实验取 L=64/128/256、m=2π/L、g₀=1±m、ξ=1/m，配合四类波形，共 24 组。外场为 gᵢ=g₀+λvᵢ，v 均值为零、RMS=1，精确 JW 参考环长 N=8L。比较 LDA、文献 GGA、NRC-LDA-R8 和 NRC-LDA，每组取 λ/m=0.0003/0.001/0.003/0.01，用前三点拟合误差阶数。各方法使用相同外场、LDA 输入与精确密度。')
    d.p('密度误差 δ=ρa−ρe 与精确密度处的 Euler 缺陷 zₐ，其领先项都由近似响应与精确响应之差决定：')
    d.eq(r'\delta=\lambda(\chi_a-\chi_e)v+O(\lambda^2),\quad z_a=\lambda(H_a\chi_e-I)v+O(\lambda^2).','delta=lambda(chi_a-chi_e)v+O(lambda^2),quad z_a=lambda(H_a chi_e-I)v+O(lambda^2).')
    rr=sorted([r for r in weak if r['L']==128 and r['side']==1 and r['family']=='mixed_scales'],key=lambda r:(list(COL).index(r['method']),r['ratio']))
    d.table('3.1','四种方法的幅度序列',['行','方法','λ/m','密度 RMS','Euler RMS','松弛差 G','梯度 ∞范数'],['line','method','ratio','density','euler','G','gradient'],[rec(r,line=record_no(r),method=r['method'],ratio=r['ratio'],density=r['density_rms'],euler=r['euler_defect_rms'],G=r['relaxation_penalty'],gradient=r['projected_gradient']) for r in rr],['i','s','g','s','s','s','s'],'L=128、正侧、混合尺度场 M，共 16 条记录；GGA 行号对应新增 CSV，其余对应原 CSV。',size=8.6)
    d.fig(3,'07_recorded_error_curves','表 3.1 的误差随幅度变化。浅色区域为拟合窗口，第四点用于检查趋势。',height=54)
    d.note('匹配均匀 Hessian 后，密度误差与 Euler 缺陷的一阶项同时消失。表中 NRC-LDA 的误差随幅度降低得更快。')

    d.page(3,'3.2　24 组实验的误差阶数')
    d.p('用误差∝λᵖ 表示衰减速度。下表的 pρ、pz、pG 分别对应密度误差、Euler 缺陷和同一泛函内的松弛能量差。全部 24 组均使用前三个幅度点拟合。')
    records=[]
    keys=sorted({(r['L'],r['side'],r['family']) for r in orders},key=lambda k:(k[0],k[1],list(FAMILY).index(k[2])))
    for i,key in enumerate(keys):
        selected=[]
        for method,metric in [('LDA','density_rms'),('GGA','density_rms'),('NRC-LDA-R8','density_rms'),('NRC-LDA','density_rms'),('NRC-LDA','euler_defect_rms'),('NRC-LDA','relaxation_penalty')]:selected.append(next(r for r in orders if (r['L'],r['side'],r['family'])==key and r['method']==method and r['metric']==metric and r['window']=='primary'))
        records.append(dict(group=f'G{i+1:02}',L=key[0],side=key[1],family=FAMILY[key[2]],pLDA=selected[0]['slope'],pGGA=selected[1]['slope'],pR8=selected[2]['slope'],pS=selected[3]['slope'],pz=selected[4]['slope'],pG=selected[5]['slope'],_refs=[r['_ref'] for r in selected]))
    d.table('3.2','24 组误差阶数',['L','侧','场','pρ LDA','pρ GGA','pρ R8','pρ NRC-LDA','pz NRC-LDA','pG NRC-LDA'],['L','side','family','pLDA','pGGA','pR8','pS','pz','pG'],records,['i','i','s','f','f','f','f','f','f'],'每行汇总 6 项拟合，原记录位置见配套清单。',size=8.45)
    d.fig(3,'02_cancellation','24 组阶数分布，黑线为中位数。NRC-LDA 接近 2、2、4；LDA、GGA 和尾部截断对照接近 1、1、2。',height=45)
    d.note('GGA 的密度误差指数为 0.99898–1.00100，Euler 缺陷仍为一阶，松弛差为二阶。增加最近邻梯度项没有消除领先线性误差；NRC-LDA 的密度与 Euler 误差为二阶，松弛差为四阶。')

    d.page(3,'3.3　截断非局域尾部后的变化')
    d.p('为检验非局域尾部的作用，只截断核在距离 R 以外的部分，并调整同位点项，使零动量修正保持为零。由均匀响应预测一阶误差系数 C₁=‖IFFT[(χR−χexact)vq]‖RMS，再用正负外场的密度误差之差测量该系数。')
    rr=sorted([r for r in tail if r['L']==128],key=lambda r:(r['side'],r['R']))
    d.table('3.3','L=128 的双侧尾部截断',['行','侧','R','R/ξ','预测 C₁','测量奇误差/λ','相对差','已分辨'],['line','side','R','scaledR','pred','measured','relative','resolved'],[rec(r,line=record_no(r),side=r['side'],R=r['R'],scaledR=r['R_over_xi'],pred=r['predicted_linear_RMS'],measured=r['measured_odd_over_lambda'],relative=r['relative_coefficient_difference'],resolved=r['linear_signal_resolved']) for r in rr],['i','i','i','g','s','s','s','s'],'两侧、9 个截断范围，共 18 行。','“未分辨”表示线性信号已接近数值噪声。',size=8.4)
    d.fig(3,'08_recorded_tail_causality','左：LDA 和 R8 的 48 组系数预测与测量；右：表 3.3 中 R>0 的结果，圆点表示已分辨测量。',height=52)
    d.note('截断到 R=8 后，一阶误差重新出现；最小幅度处，系数预测与测量的最大相对差为 7.56×10⁻⁷。保留非局域尾部是消除这一误差的关键。')

    d.page(3,'3.4　密度剖面与松弛能量差')
    profiles,sid=b.json(ROOT/PARTS[2]/'profiles/weak_field_profiles.json');idx=next(i for i,r in enumerate(profiles) if r['L']==128 and r['side']==1 and r['family']=='mixed_scales' and r['method']=='NRC-LDA' and r['ratio']==.003);p=profiles[idx]
    records=[dict(site=i,exact=p['exact_plus'][i],NRC_LDA=p['rho_plus'][i],difference=p['rho_plus'][i]-p['exact_plus'][i],Euler=p['euler_defect'][i],_refs=[dict(source=sid,kind='json',index=idx,site=i,fields=['exact_plus','rho_plus','euler_defect'])]) for i in range(0,128,8)]
    d.table('3.4','16 个等间距格点的密度',['格点 i','精确密度','NRC-LDA 密度','两密度之差','Euler 缺陷'],['site','exact','NRC_LDA','difference','Euler'],records,['i','rho','rho','s','s'],'背景同表 3.1；NRC-LDA、λ/m=0.003，i=0,8,…,120。','密度保留 10 位小数，差值用原始精度计算；完整剖面见 JSON。',size=9)
    d.p(f'全 128 个格点的密度误差 RMS={p["density_rms"]:.6e}，Euler 缺陷 RMS={p["euler_defect_rms"]:.6e}，投影梯度={p["projected_gradient"]:.6e}。表 3.5 比较同组四个幅度下的能量差计算。')
    rr=sorted([r for r in weak if r['L']==128 and r['side']==1 and r['family']=='mixed_scales' and r['method']=='NRC-LDA'],key=lambda r:r['ratio'])
    d.table('3.5','松弛能量差的计算对照',['行','λ/m','G：8 点积分','G：4 点积分','直接能量相减','G/二次型'],['line','ratio','GL8','GL4','direct','quadratic_ratio'],[rec(r,line=record_no(r),ratio=r['ratio'],GL8=r['relaxation_penalty'],GL4=r['penalty_GL4'],direct=r['penalty_direct_subtraction'],quadratic_ratio=r['penalty_over_quadratic']) for r in rr],['i','g','s','s','s','f'],'同一组的四个幅度。')
    d.eq(r'G_a=[E_a(\rho_e)-E_a(\rho_a)]/L\simeq\delta^T H_a\delta/(2L).','G_a=[E_a(rho_e)-E_a(rho_a)]/L approx delta^T H_a delta/(2L).')
    replay=json.loads((ROOT/PARTS[2]/'logs/reproduction_check.json').read_text())
    d.p('G 是同一近似泛函在精确密度与优化密度之间的能量差。它随密度误差的平方变化，因此表现为四阶；这里的四阶指松弛差，而非真实基态能量误差。')
    d.note(f'微小能量差用梯度路径积分计算。4 点与 8 点积分一致，直接相减在弱场下会损失精度。独立重算 288 条结果，密度误差的最大差异为 {replay["maximum_density_error_difference"]:.3e}。')

    d.page(4,'4.1　固定扰动幅度下的临界逼近')
    d.p('消除一阶误差后，剩余误差在接近临界点时有多大？实验沿两条路径考察：固定 L=32，将 m 从 0.1 降至 0.0001；或取 L=64/128/256，保持 q/m=1。外场为 gᵢ=g₀+λvᵢ，v 均值为零、RMS=1，不同 L 使用相同无量纲波形，参考解采用热力学 Bloch/JW 计算。')
    d.p('将密度误差除以外场诱导的密度变化，得到相对误差 ερ；α 表示峰值扰动与临界距离之比。')
    d.eq(r'\epsilon_\rho=\frac{\|\rho_{NRC-LDA}-\rho_e\|_{RMS}}{\|\rho_e-\rho_{hom}(g_0)\|_{RMS}},\quad\alpha=\lambda\|v\|_\infty/m.','epsilon_rho=frac(norm(rho_("NRC-LDA")-rho_e)_("RMS"),norm(rho_e-rho_("hom")(g_0))_("RMS")),quad alpha=lambda norm(v)_infinity/m.')
    rr=sorted([r for r in scan if r['protocol']=='fixed_period' and r['amplitude_kind']=='absolute' and r['lambda']==.001],key=lambda r:(-r['m'],r['side'],r['family']))
    comparison=[]
    for r in rr:
        c=next(x for x in gga_critical if x['id']==r['id'])
        comparison.append(dict(line=record_no(r),m=r['m'],side=r['side'],family=FAMILY[r['family']],signal=r['signal_rms'],error=r['error_rms'],epsilon=r['relative_error'],gga=c['relative_error'],lda=c['lda_relative_error'],_refs=ref(r)+ref(c)))
    d.table('4.1','固定 L=32、λ=0.001 的同场对照',['行','m','侧','场','诱导 RMS','NRC 误差 RMS','NRC ε (%)','GGA ε (%)','LDA ε (%)'],['line','m','side','family','signal','error','epsilon','gga','lda'],comparison,['i','s','i','s','s','s','p','p','p'],'两侧、两波形，共 28 行；NRC=NRC-LDA。行号对应原扫描，GGA 记录按相同外场 id 配对。','三种方法均用同一诱导密度 RMS 归一化。',size=8.4)
    d.note('m 从 0.1 降至 0.0001 时，GGA 最坏相对误差从 23.47% 增至 169.57%，LDA 从 67.84% 增至 328.13%；NRC-LDA 从 0.0855% 增至 6.479%。GGA 改善了 LDA，但未恢复临界响应；NRC-LDA 的剩余非线性误差也随临界逼近增大。')

    d.page(4,'4.2　1% 和 5% 误差下的适用窗口')
    d.p('以 1% 和 5% 为误差标准，从小到大增加扰动幅度，再加密通过与未通过点之间的区间。求解同时满足梯度∞范数<10⁻⁹、背景有效、未触边界且保持原背景侧。表 4.2 汇总两侧、两波形的共同窗口：λ通过为四组都通过的上端点，λ失败为首次有组未通过的幅度。')
    rr=sorted(env,key=lambda r:(r['protocol'],-r['m'],r['tolerance']))
    d.table('4.2','20 个共同适用窗口',['协议','L','m','容差 (%)','λ通过','λ失败','α通过','右删失组数'],['protocol','L','m','tolerance','pass','fail','alpha','censored'],[rec(r,protocol='固定 L' if r['protocol']=='fixed_period' else 'q/m=1',L=r['L'],m=r['m'],tolerance=r['tolerance'],**{'pass':r['lambda_verified_all'],'fail':r['lambda_first_failure_any']},alpha=r['alpha_verified_all'],censored=r['right_censored']) for r in rr],['s','i','s','p','s','s','g','i'],'每行汇总两侧、两波形的四个窗口。','右删失表示扫描结束时该组仍通过。',size=8.4)
    d.fig(4,'05_verified_windows','共同窗口随临界距离的变化。实线为 q/m=1，虚线为固定 L；λ 与 α 分别取四组中的最小通过值。',height=50)
    d.note('固定 L=32、m=10⁻⁴ 时，1% 窗口的通过/未通过幅度为 [1.6261×10⁻⁴,1.7437×10⁻⁴]；5% 为 [8.1092×10⁻⁴,8.6960×10⁻⁴]。这些数值给出了所测波形下的使用范围。')

    d.page(4,'4.3　窗口边界的数值检查')
    selected=[]
    for label,L,m in [('A',32,.0001),('B',256,2*np.pi/256)]:
        for tol in [.01,.05]:
            w=next(r for r in windows if r['L']==L and np.isclose(r['m'],m,rtol=1e-12) and r['side']==1 and r['family']=='localized_packet' and r['tolerance']==tol)
            for suffix,key in [('通','limit_id'),('失','failure_id')]:
                r=next(r for r in scan if r['id']==w[key]);selected.append((f'{label}{int(tol*100)}{suffix}',r,tol))
    d.p('选取两组边界检查计算精度：A 为 L=32、m=10⁻⁴，B 为 L=256、q/m=1；均取正侧波包 P。表 4.3 列出两档容差的相邻通过/未通过点。点名数字表示容差百分比，“通/失”表示判定结果。')
    d.table('4.3','两条路径的边界记录',['点','源行','λ','α','ερ (%)','含裕量 ε (%)','梯度 ∞范数'],['point','line','lambda','alpha','epsilon','upper','gradient'],[rec(r,point=label,line=record_no(r),**{'lambda':r['lambda']},alpha=r['alpha'],epsilon=r['relative_error'],upper=r['relative_error_upper'],gradient=r['gradient']) for label,r,t in selected],['s','i','s','g','p','p','s'],'A、B 两组，各列两档容差的相邻端点。')
    control_records=[]
    for label,r,t in selected:
        c=next(x for x in bc if x['id']==r['id']);control_records.append(rec(c,point=label,line=record_no(c),refined=c['relative_error_refined'],shift=c['refinement_shift_over_signal'],eigen=c['minimum_Hessian'],restart=c['alternative_start_density_difference'],same=c['classification_1pct_agrees'] and c['classification_5pct_agrees']))
    d.table('4.4','边界点的核表加密与初值检查',['点','控制源行','加密 ε (%)','加密位移/信号','最小 Hessian','换初值密度差','分类相同'],['point','line','refined','shift','eigen','restart','same'],control_records,['s','i','p','s','f','s','s'],'各点与表 4.3 一一对应。')
    failed=[r for r in scan if not r['solver_valid']]
    d.table('4.5','未达求解精度要求的记录',['源行','m','侧/场','λ','ερ (%)','梯度 ∞范数','solver valid'],['line','m','case','lambda','epsilon','gradient','valid'],[rec(r,line=record_no(r),m=r['m'],case=f'{int(r["side"]):+d}/{FAMILY[r["family"]]}',**{'lambda':r['lambda']},epsilon=r['relative_error'],gradient=r['gradient'],valid=r['solver_valid']) for r in failed],['i','s','s','s','p','s','s'],'该点未计入通过窗口。')
    d.p(f'共检查 157 个边界点。加密核表后，密度变化与信号之比最大为 {control["max_refinement_shift_over_signal"]:.5e}，通过/未通过判定均未改变。48 个更高精度积分检查的最大比值为 {control["max_quadrature_shift_over_signal"]:.5e}；均匀响应与独立积分的最大相对差为 {control["max_uniform_quadrature_difference"]:.5e}。')
    d.p(f'28 个 GGA 对照另做均匀插值表加密与初值检查：密度变化/信号最大 {gga_summary["max_refinement_shift_over_signal"]:.3e}，换初值密度差最大 {gga_summary["max_restart_density_difference"]:.3e}。')
    d.note('加密核表、提高积分精度和更换初值后，窗口边界保持稳定。边界点的最小 Hessian 特征值均为正；表 4.5 的一个点未达求解精度要求，未计入通过窗口。')

    d.page(4,'4.4　剩余误差与修复的局限')
    d.p('若密度误差≈C₂λ²、诱导响应≈A₁λ，则 ερ≈λC₂/A₁。即使误差保持二阶，C₂ 增大仍会缩小适用窗口。拟合使用至少三个信号超过数值噪声 20 倍的点，点数不足时记为“—”。')
    rr=sorted([r for r in coeff if r['protocol']=='fixed_period' and r['side']==1 and r['family']=='localized_packet'],key=lambda r:-r['m'])
    d.table('4.6','正侧波包的二阶误差系数',['源行','m','已分辨点数','误差指数','C₂','线性信号系数'],['line','m','count','slope','C2','A1'],[rec(r,line=record_no(r),m=r['m'],count=r['points'],slope=r.get('slope'),C2=r.get('C2'),A1=r.get('linear_signal_coefficient')) for r in rr],['i','s','i','f','s','s'],'固定 L=32、正侧波包，列出全部 7 个 m。')
    d.fig(4,'06_nonlinear_coefficient','临界逼近时 C₂ 增大，误差阶数仍接近 2。40 组中 30 组满足拟合条件；窗口边界由直接误差确定。',height=46)
    nl=[]
    for period in sorted({r['period'] for r in nonlinear}):
        a=next(r for r in nonlinear if r['period']==period and r['method']=='JW');c=next(r for r in nonlinear if r['period']==period and r['method']=='Oracle');nl.append(dict(period=period,exact1=a['A1'],app1=c['A1'],exact2=a['A2'],app2=c['A2'],_refs=ref(a)+ref(c)))
    d.table('4.7','四个周期的高阶响应',['周期','精确 A₁','NRC-LDA A₁','精确 A₂','NRC-LDA A₂'],['period','exact1','app1','exact2','app2'],nl,['i','f','f','f','f'],'N=128、g₀=0.9，单模场；A₁/A₂ 为基频/二次谐波系数，取三个幅度点按 λ² 外推。','旧名称 Oracle 与 NRC-LDA 同式。')
    br=[]
    for optimizer in ['L-BFGS-B','SLSQP']:
        for cluster in [0,1]:
            group=[r for r in branches if r['optimizer']==optimizer and r['cluster']==cluster];r=min(group,key=lambda x:x['energy']);br.append(rec(r,optimizer=optimizer,cluster=cluster,count=len(group),energy=r['energy'],gradient=r['projected_gradient'],hessian=r['hessian_min']))
    d.table('4.8','两种优化器得到的稳定分支',['优化器','簇','簇内点数','总能量','投影梯度','最小 Hessian'],['optimizer','cluster','count','energy','gradient','hessian'],br,['s','i','i','rho','s','f'],'N=32、g₀=1、幅度 0.4；每个优化器取 24 个初值，按解分组后列出各组最低能量。',size=8.5)
    d.note('NRC-LDA 修复了二点响应，但二次谐波仍有偏差，强场下也可出现多个稳定解。窗口实验尚未覆盖 m=0，q/m=1 路径最低到 m≈0.02454。80 个窗口端点中，51 个出现局域 g 跨过 1，但均未达到全链闭隙条件 ∏gᵢ=1。')

    d.page(0,'数据索引与复核方法')
    d.p('正文表格由数据文件自动生成。下列 S 编号对应表下注释；原始行号、数组索引和完整哈希见 table_provenance.json。')
    source_records=[]
    for s in b.sources.values():
        pp=Path(s['path']);section=next((str(i+1) for i,p in enumerate(PARTS) if pp.parts[0]==p),'—');source_records.append(dict(id=s['id'],section=section,file=pp.name,hash=s['sha256'][:12],_refs=[]))
    # The source index is metadata rather than an experimental table.
    d.table('D.1','正文数据源索引',['源','类别','数据文件','SHA-256 前 12 位'],['id','section','file','hash'],source_records,['s','s','s','s'],'来源标识和完整哈希见 table_provenance.json。',size=8.2)
    rawrows=sum(len(t['records']) for t in b.tables if t['id']!='D.1')
    d.p(f'正文共 {len(b.tables)-1} 张数据表、{rawrows} 行记录，包括有限范围重拟合对照、24 组误差阶数、28 条固定幅度扫描和 20 个共同窗口。完整数据保存在各子项目中。')
    d.p('report/data_tables/ 保存各表的完整精度 CSV；table_provenance.json 记录选样规则和数据位置。可编辑正文、Typst 源文件与 PDF 同步提供。')
    d.p('在 TFIM 根目录运行 python run.py --stages report audit 可重建报告并核验数据；python -m tfim.weak_field 可重算第三部分；python run.py 可重算临界窗口。')
    d.save()
    after={str(p.relative_to(ROOT)):sha(p) for p in scientific};assert before==after
    doc=fitz.open(REPORT/'TFIM_核心研究与临界适用窗口_含重拟合对照.pdf');counts={i:sum(p.get_text().splitlines()[0]==SECTION[i] for p in doc) for i in SECTION}
    save(REPORT/'report_layout_validation.json',dict(planned_pages=len(d.pages),actual_pages=len(doc),section_pages=counts,scientific_tables=len(b.tables)-1,scientific_table_rows=rawrows,source_count=len(b.sources),scientific_files_unchanged=True,scientific_input_hashes=before,section_3_and_4_balanced=counts[3]==counts[4],source_table_manifest_sha256=sha(REPORT/'table_provenance.json')))
    print('Balanced report built:',len(doc),'pages; section counts',counts,';',rawrows,'scientific table rows',flush=True)
