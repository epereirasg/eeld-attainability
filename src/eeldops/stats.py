"""Statistics used in the paper: bootstrap CI, paired Wilcoxon, Holm, Hodges-Lehmann, rank-biserial, mean +- SD."""
import numpy as np
from scipy.stats import wilcoxon, spearmanr


def boot_ci(x, stat=np.mean, B=10000, seed=0, level=0.95):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    s = np.array([stat(rng.choice(x, len(x))) for _ in range(B)])
    a = (1 - level) / 2
    return tuple(np.quantile(s, [a, 1 - a]))


def mean_sd(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else np.nan


def hodges_lehmann(d):
    d = np.asarray(d, float); i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def rank_biserial(d):
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0: return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired(a, b, alternative='two-sided'):
    """Paired comparison a vs b: n, means, SDs, HL shift, rank-biserial, Wilcoxon p, bootstrap CI of mean difference."""
    a = np.asarray(a, float); b = np.asarray(b, float); m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    d = a - b
    p = wilcoxon(d, alternative=alternative, zero_method='wilcox').pvalue if np.any(d != 0) else 1.0
    return dict(n=int(len(d)), mean_a=a.mean(), sd_a=a.std(ddof=1), mean_b=b.mean(), sd_b=b.std(ddof=1),
                mean_diff=d.mean(), ci_lo=boot_ci(d)[0], ci_hi=boot_ci(d)[1], HL=hodges_lehmann(d), rbc=rank_biserial(d), p=p)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); k = len(p); adj = np.empty(k); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (k - r) * p[i])); adj[i] = run
    return adj


def spearman(x, y):
    r = spearmanr(x, y); return float(r.statistic), float(r.pvalue)
