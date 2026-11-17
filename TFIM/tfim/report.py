import json,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .common import ROOT,NEW,PARTS,read,write,save

COL={'LDA':'#7a818b','NRC-LDA-R8':'#a34f79','NRC-LDA':'#087e91'}
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
    fig,axs=plt.subplots(1,3,figsize=(10.4,2.9),layout='constrained');rr=sorted(data(0,'critical_kernel'),key=lambda r:r['q']);q=np.array([r['q'] for r in rr]);k=np.array([r['K_exact'] for r in rr]);mask=q<.1;axs[0].semilogx(q[mask],k[mask]*np.log(1/q[mask])/np.pi,color=COL['NRC-LDA']);axs[0].axhline(1,ls='--',color='k',lw=.8);axs[0].set(xlabel='$q$',ylabel=r'$K(q) \log(1/q)/\pi$',title='Critical nonanalyticity')
    rr=data(0,'critical_crossover');ss=sorted(set(r['s'] for r in rr));axs[1].semilogx(ss,[next(r['theory'] for r in rr if r['s']==s) for s in ss],'k-',label='limit')
    for m,c in [(1e-4,'#d49932'),(1e-8,COL['NRC-LDA'])]:
        for side,ls in [(-1,'--'),(1,'-')]:
            a=sorted([r for r in rr if r['m']==m and r['side']==side],key=lambda r:r['s']);axs[1].semilogx([r['s'] for r in a],[r['exact_scaled'] for r in a],ls,color=c,label=f'$m={m:g}$' if side==1 else None)
    axs[1].set(xlabel='$q/m$',ylabel='scaled kernel difference',title='Nonpolynomial crossover');axs[1].legend(fontsize=7)
    rr=data(0,'bounded_range_scaling')
    for R,c in [(1,COL['LDA']),(8,COL['NRC-LDA-R8'])]:
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
    from .workspace import directory_notes
    directory_notes()

def run():
    figures()
    from .report_layout import build
    build()
    documents()
    print('Four-part report and focused directory documentation generated',flush=True)
