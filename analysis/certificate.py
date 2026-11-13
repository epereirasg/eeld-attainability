"""RQ4: solver certificate summary."""
import glob
import json
import pandas as pd
from common import *
CERT = str(paths.RAW_CERT)
N = {}
for case, T in [('10', 'Ten'), ('40', 'Forty')]:
    s = glob.glob(f'{CERT}/structural_e{case}.csv')
    if s:
        S = pd.read_csv(s[0])
        N[f'CertCells{T}'] = f'{int(S.cells.sum()):,}'.replace(',', '{,}')
        N[f'CertViol{T}'] = str(int(S[['viol_R', 'viol_rho', 'viol_frontier', 'viol_phys', 'out_of_range']].sum().sum()))
    d = glob.glob(f'{CERT}/de_e{case}_p*.csv')
    if d:
        D = pd.concat([pd.read_csv(f) for f in d])
        N[f'CertSample{T}'] = str(len(D)); N[f'CertMaxImp{T}'] = f'{D.improvement.max():.4f}'
        N[f'CertFail{T}'] = str(int((D.improvement > 0.01).sum()))
        N[f'CertCampaignBetter{T}'] = f'{100*(D.improvement < -0.001).mean():.0f}'
        D.to_csv(f'{PROC}/certificate_de_e{case}.csv', index=False)
with open(f'{MAC}/numbers_cert.tex', 'w') as f:
    for k, v in sorted(N.items()): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
json.dump(N, open(f'{PROC}/numbers_cert.json', 'w'), indent=1)
print(json.dumps(N, indent=1))
