import json,csv,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PARTS=['01_No-Go_Theorems','02_Minimal_Nonlocal_Correction','03_Error-Cancellation_Mechanism','04_Critical_Applicability_and_Limitations']
NEW=ROOT/PARTS[3]
CONFIG=json.loads((NEW/'config/experiment.json').read_text(encoding='utf8'))
WORK_DIRS=('config','data','profiles','cache','figures','logs')
def init():
    for part in PARTS:
        for folder in WORK_DIRS:(ROOT/part/folder).mkdir(parents=True,exist_ok=True)
    (ROOT/'logs').mkdir(exist_ok=True)
    (ROOT/'report').mkdir(exist_ok=True)
def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
def read(path):
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
    for r in rows:
        for k,v in r.items():
            if v=='':r[k]=None
            elif v in ['True','False']:r[k]=v=='True'
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows
def write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def masses(L):return CONFIG['fixed_period_masses'] if L==CONFIG['fixed_period'] else [2*np.pi/L]
def fields(L):
    # Keep the same dimensionless waveform at every L; only the spatial scale changes.
    theta=2*np.pi*np.arange(L)/L;rng=np.random.default_rng(CONFIG['seed']+CONFIG['fixed_period']);out=[]
    modes=np.arange(1,5);w=rng.uniform(.5,1.5,4)/np.sqrt(modes);ph=rng.uniform(-np.pi,np.pi,4)
    for family in CONFIG['families']:
        v=np.sum(w[:,None]*np.cos(modes[:,None]*theta+ph[:,None]),axis=0) if family=='four_low_modes' else np.exp(2*np.cos(theta-.73))*np.cos(3*theta+.41)
        v-=v.mean();v/=rms(v);out.append(dict(L=L,family=family,v=v.tolist(),peak=float(max(abs(v)))))
    return out
