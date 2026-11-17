import json
from pathlib import Path
from ..common import ROOT,PARTS,rms,read as read_path,write as write_path,save
BASE=ROOT/PARTS[2]
CONFIG=json.loads((BASE/'config/experiment.json').read_text(encoding='utf8'))
METHODS=CONFIG['methods']
RESULTS=BASE/'cache/replay'
def init():
    for d in ['raw','profiles','cache']:(RESULTS/d).mkdir(parents=True,exist_ok=True)
def write(name,rows):init();write_path(RESULTS/'raw'/f'{name}.csv',rows)
def read(name):return read_path(RESULTS/'raw'/f'{name}.csv')
def jsave(name,obj,folder='raw'):init();save(RESULTS/folder/f'{name}.json',obj)
def jload(name,folder='raw'):return json.loads((RESULTS/folder/f'{name}.json').read_text(encoding='utf8'))
