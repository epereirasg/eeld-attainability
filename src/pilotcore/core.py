"""Pilot dispatch core used by the method comparison (experiments/robustness, experiments/methods_dfo).

Frozen copy of the pilot-study dispatch core. Only the data paths changed: the coefficient files are the
repository copies, which are byte-identical (same SHA-256) to the files of the earlier eeld-cf
repository that the pilot read. The numerical code is unchanged so that the stored method-comparison
runs remain reproducible. New code should use ``eeldattain.model`` instead.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize, differential_evolution
from eeldattain.paths import COEFFS
D10 = str(COEFFS['10'])
D40 = str(COEFFS['40'])
class Sys:
    def __init__(s,path,escale=1.0):
        d=pd.read_csv(path); s.n=len(d); s.lo=d.Pmin.values.astype(float); s.hi=d.Pmax.values.astype(float)
        s.a,s.b,s.c,s.e,s.f=[d[k].values.astype(float) for k in 'abcef']
        s.al,s.be,s.ga,s.ze,s.la=[d[k].values.astype(float) for k in ['alpha','beta','gamma','zeta','lambda']]
        s.k=escale; s.rng_=s.hi-s.lo
    def C(s,P): return float(np.sum(s.a*P**2+s.b*P+s.c+np.abs(s.e*np.sin(s.f*(s.lo-P)))))
    def E(s,P): return float(np.sum(s.k*(s.al+s.be*P+s.ga*P**2)+s.ze*np.exp(s.la*P)))
    def F(s,P): return np.array([s.C(P),s.E(P)])
    def gC(s,P):
        z=s.f*(s.lo-P); return 2*s.a*P+s.b+np.sign(s.e*np.sin(z))*s.e*np.cos(z)*(-s.f)
    def gE(s,P): return s.k*(s.be+2*s.ga*P)+s.ze*s.la*np.exp(s.la*P)
    def rand_feas(s,PD,rng):
        for _ in range(1000):
            P=rng.uniform(s.lo,s.hi); 
            for _ in range(50):
                r=PD-P.sum(); 
                if abs(r)<1e-9: break
                room=(s.hi-P) if r>0 else (P-s.lo); 
                if room.sum()<=0: break
                P=P+r*room/room.sum(); P=np.clip(P,s.lo,s.hi)
            if abs(P.sum()-PD)<1e-6: return P
        raise RuntimeError
def nlp(sysm,PD,fun,jac,x0s,lo=None,hi=None,cons=(),tol=1e-6):
    lo=sysm.lo if lo is None else lo; hi=sysm.hi if hi is None else hi
    C=[{'type':'eq','fun':lambda P:P.sum()-PD,'jac':lambda P:np.ones_like(P)},*cons]
    best=None
    for x0 in x0s:
        r=minimize(fun,np.clip(x0,lo,hi),jac=jac,method='SLSQP',bounds=list(zip(lo,hi)),constraints=C,options={'maxiter':2000,'ftol':1e-12})
        P=r.x
        ok=abs(P.sum()-PD)<1e-5 and np.all(P>=lo-1e-7) and np.all(P<=hi+1e-7) and all(c['fun'](P)>=-tol for c in cons)
        if ok and (best is None or fun(P)<best[0]): best=(fun(P),P.copy())
    return best
def de_global(sysm,PD,fun,lo=None,hi=None,pen_cons=(),seed=0,maxiter=600,pop=30):
    """DE in reduced space: last unit is slack; penalties for slack bounds and extra constraints."""
    lo=sysm.lo if lo is None else lo; hi=sysm.hi if hi is None else hi; n=sysm.n
    def full(z): return np.append(z,PD-z.sum())
    def obj(z):
        P=full(z); v=max(0,lo[-1]-P[-1])+max(0,P[-1]-hi[-1])
        for g in pen_cons: v+=max(0,-g(P))
        return fun(np.clip(P,lo,hi))+1e7*v
    r=differential_evolution(obj,list(zip(lo[:-1],hi[:-1])),seed=seed,maxiter=maxiter,popsize=pop,tol=1e-10,polish=False,init='sobol')
    return full(r.x)
