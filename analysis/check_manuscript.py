"""Check that a manuscript source carries the numbers, tables and figures of this repository.

Usage: python analysis/check_manuscript.py PATH/TO/main.tex

The manuscript is a single LaTeX file that includes the figures of results/figures/ from its own folder. It may inline
the number macros (results/macros/numbers_*.tex) and the generated tables (results/tables/*.tex) between marker
comments, in which case every inlined block is compared with the repository copy. A manuscript with the values typed
out is checked in two other ways. Every generated table that the manuscript reproduces (three or more of its rows
appear verbatim) must appear in full, and every decimal number of the text, the captions and the tables must equal a
value of a number macro or of a generated table. The script exits with an error on any difference. Run it after `python run_experiments.py --exp tables` and `scripts/make_figures.sh`.
Figure files are compared byte by byte, so a figure redrawn on another platform is reported as redrawn, not as an error.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAC, TAB, FIG = ROOT / 'results' / 'macros', ROOT / 'results' / 'tables', ROOT / 'results' / 'figures'
BLOCK = re.compile(r'% ---- begin (macros|tables)/(\S+) \(inlined\)\n(.*?)% ---- end \1/\2', re.S)
DEF = re.compile(r'\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$')
NUM = re.compile(r'(?<![\w.,\\])-?\d{1,3}(?:,\d{3})*\.\d+(?![\d.]*\s*(?:cm|pt|em|ex|mm|in|\\columnwidth|\\textwidth)\b)')
# values of the text that are inputs rather than outputs: the design (weights, cuts, tolerances, band grid, the
# bisection step 1000/256 MW), half the 25 MW budget, the residual bound sqrt(2) * 0.05 of Proposition 1 and the
# software versions
LITERAL = {'0.25', '0.5', '0.75', '2.5', '0.1', '0.02', '0.05', '0.01', '0.10', '0.20', '3.9', '12.5', '7.1', '3.13', '6.3'}


def flat(s):
    return ' '.join(s.split())


def body(text):
    """Abstract to bibliography, without TikZ drawings, whose coordinates are not results."""
    i, j = text.find('\\begin{abstract}'), text.find('\\bibliography')
    b = text[i if i >= 0 else 0: j if j >= 0 else len(text)]
    return re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', ' ', b, flags=re.S)


def typed_out(text, bad):
    """Checks for a manuscript that types the values out instead of inlining the macro files."""
    stored = ''.join(f.read_text() for f in sorted(MAC.glob('numbers_*.tex')))
    tabs = {f.name: f.read_text() for f in sorted(TAB.glob('*.tex'))}
    ft = flat(text); n_tab = 0
    for name, src in tabs.items():
        rs = [flat(r) for r in src.splitlines() if '&' in r and r.strip().endswith('\\\\')]
        hit = [r in ft for r in rs]
        if sum(hit) >= 3:
            n_tab += 1
            bad += [f'table {name}: row missing from the manuscript: {r[:70]}' for r, h in zip(rs, hit) if not h]
    vals = set(NUM.findall((stored + ''.join(tabs.values())).replace('{,}', ',')))
    vals |= {v.lstrip('-') for v in vals}
    nums = NUM.findall(body(text).replace('{,}', ','))
    for v in sorted(set(nums)):
        if v not in vals and v.lstrip('-') not in vals and v not in LITERAL:
            bad.append(f'number {v} of the manuscript is not a stored macro or table value')
    return n_tab, len(nums)


def macros(text):
    out = {}
    for line in text.splitlines():
        m = DEF.match(line.strip())
        if m:
            out.setdefault(m.group(1), m.group(2))
    return out


def rows(text):
    return [l.strip() for l in text.strip().splitlines() if l.strip()]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tex = Path(sys.argv[1]); text = tex.read_text()
    bad = []; n_mac = n_tab = 0
    for kind, name, body in BLOCK.findall(text):
        if kind == 'macros':
            f = MAC / name
            if not f.exists():
                continue
            have, want = macros(body), macros(f.read_text())
            for k, v in have.items():
                n_mac += 1
                if want.get(k) != v:
                    bad.append(f'macro \\{k} of {name}: manuscript {v!r}, repository {want.get(k)!r}')
        elif (TAB / name).exists():
            n_tab += 1
            if rows(body) != rows((TAB / name).read_text()):
                bad.append(f'table {name} differs from results/tables/{name}')
        elif name.startswith('tab_ablation_'):          # the two per-system tables drop the system column of tab_ablation.tex
            n_tab += 1
            src = ' '.join((TAB / 'tab_ablation.tex').read_text().split())
            for r in rows(body):
                if '&' in r and not r.startswith('Instr.') and ' '.join(r.split()) not in src:
                    bad.append(f'table {name}: row not in results/tables/tab_ablation.tex: {r[:60]}')
    figs = sorted(set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+\.pdf)\}', text)))
    redrawn = []
    for f in figs:
        a, b = tex.parent / f, FIG / f
        if not b.exists():
            bad.append(f'figure {f} is not in results/figures/')
        elif a.exists() and a.read_bytes() != b.read_bytes():
            redrawn.append(f)
    if n_mac == 0 and n_tab == 0:
        n_tab, n_num = typed_out(text, bad)
        print(f'{n_num} typed-out numbers, {n_tab} generated tables and {len(figs)} figures checked')
    else:
        print(f'{n_mac} inlined macros, {n_tab} inlined tables and {len(figs)} figures checked')
    if redrawn:
        print('figures redrawn since the manuscript was built (compare visually):', ', '.join(redrawn))
    if bad:
        print('\n'.join(bad)); sys.exit(f'{len(bad)} differences')
    print('the manuscript matches the repository')


if __name__ == '__main__':
    main()
