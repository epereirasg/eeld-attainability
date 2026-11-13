# Input data

Every input of the study is a small CSV file in this folder. All files are included in the repository,
and nothing has to be downloaded. The loaders check the SHA-256 sums below at every run and stop if a
file has changed (`SHA256SUMS.txt` for the benchmark tables, `EXPECTED_SHA256` in
`src/eeldops/data_loader.py` for the external tables).

## Files

| File | Content | Source | SHA-256 |
|---|---|---|---|
| `eeld10_coeffs.csv` | 10-unit generator limits, quadratic and valve-point cost coefficients (`a`, `b`, `c`, `e`, `f`) and exponential emission coefficients (`alpha`, `beta`, `gamma`, `zeta`, `lambda`) | Basu, Int. J. Electr. Power Energy Syst. 30(2), 2008, as tabulated by Pandit et al., Appl. Soft Comput. 12(11), 2012 | `ea9ff88e46de8ee7e18c4d8ca542c476a35129f3e08857397a7ea8006c9aa42c` |
| `demand_eeld10.csv` | 24-hour demand profile, 1036 to 2220 MW, peak in hour 12 | Dynamic dispatch profile of Attaviriyanupap et al., IEEE Trans. Power Syst. 17(2), 2002 | `c9970154f4be58460e3498f53e7a401eee9d3764c714b59ed1eb6ee4f7aad6b2` |
| `eeld40_coeffs.csv` | 40-unit valve-point benchmark, same columns as the 10-unit file | Sinha et al., IEEE Trans. Evol. Comput. 7(1), 2003 (cost data), with the emission coefficients tabulated by the Al-Roomi Economic Load Dispatch Test Systems Repository (2016) | `a8293dbdb17d36d2f92d2877162f1bbe3dab25e21c21962927db6e90e2f8c235` |
| `demand_eeld40.csv` | Six demand levels, 8000 to 12000 MW. These are stress-test levels, not a time series | Stress-test levels chosen by the authors | `891f6f9060e00d70f9adf18b1693297993f316ec7d514d951fa65cd12a9edcb6` |
| `external/eeld10_B_matrix.csv` | 10 by 10 Kron loss matrix (1/MW) | `deed_env.py` of github.com/danielcregg/jupyterNotebooks, commit `7f451626580cd7fa91bf6c4ade3c22bead469cd2` (20 Feb 2026) | `091a2d12945c6830881b72d03d5d683815538dd595e4c1a4959619a94e0a9c5d` |
| `external/eeld10_ramp_limits.csv` | Up and down ramp rates (MW/h) with the unit limits | Same file and commit | `68c553b6080a905d1d1d73ef3bf5789796d0ffe3f69131ce6dd2777beede4e67` |
| `external/eeld10_gen_external.csv` | Generator table of the same source, used only for the identity check with `eeld10_coeffs.csv` | Same file and commit | `643289aa03e8abb873415d3c9e13cdf4ac60d43c46ef7347989b6c08cc6262cf` |

The three files in `external/` were extracted from the `GEN_DATA` and `B_MATRIX` arrays of that file. They are
redistributed under the MIT licence of the source repository, whose notice is in `external/LICENSE`.

## Checks

* The external generator table agrees with `eeld10_coeffs.csv` in all 12 shared columns.
  `src/eeldops/data_loader.py` repeats this check, and checks that the loss matrix is symmetric and positive
  definite, at every load.
* The ramp rates are 80 MW/h for units 1 to 3, 50 MW/h for units 4 to 6 and 30 MW/h for units 7 to 10.
* The manuscript reports that the coefficients, the ramp rates and the loss matrix agree with Table A.1 and
  Appendix B of Basu (2008) and with Tables A5 and A6 of Pandit et al. (2012). The published tables are not
  redistributed here, so the loaders do not repeat this comparison.

## Emission convention

The emission function is E_i = alpha_i + beta_i P_i + gamma_i P_i^2 + zeta_i exp(lambda_i P_i) (lb/h). Eq. (2) of
Pandit et al. (2012) multiplies the quadratic part by 1e-2. `System(..., escale=0.01)` reproduces that convention,
and the method comparison includes it as a sensitivity configuration. `experiments/auxiliary/daily_optima.py`
writes the daily optima under both conventions (`results/raw/daily_optima.json`), which Appendix B of the
manuscript uses to justify the choice.

The 40-unit source prints the same factor. The minimum-emission optimum that Bhattacharjee et al., Int. J. Electr.
Power Energy Syst. 59, 2014, report for this system at 10,500 MW agrees with the form without the factor, so the
primary 40-unit campaign uses `escale=1`. The variant with the printed factor (`escale=0.01`) is stored in
`results/raw/additional/forty_printed/`. Every 40-unit script reads the factor of the stored campaign from its
configuration file (`eeldattain.runio.campaign_escale`).

## Stylized 40-unit operational data

No loss matrix or ramp table is published for the 40-unit system. The 40-unit operational replication
(`experiments/supplementary/ops40.py`) builds both from two stated assumptions, and every output row records
them. Ramp rates are proportional to the unit span with the median 10-unit ratio (0.297 per hour). The loss
matrix consists of four copies of the 10-unit Kron matrix, scaled so that the median incumbent loss share
matches the 10-unit value of about 3% of demand. These are modelling assumptions, not benchmark data.

## 30-bus network case

| File | Source | SHA-256 |
|---|---|---|
| `eeld30_coeffs.csv` | Table I of M. A. Abido, IEEE Trans. Power Syst. 18(4):1529-1537, 2003 (doi 10.1109/TPWRS.2003.818693), read from the publisher PDF and converted from p.u. on 100 MVA to MW. Columns `Pmin`, `Pmax` and `bus` are assumptions (see below) | e08da206b2210a8f14d61ae3b68858020f5f3cae00a54af72bff5bfd6afede78 |
| `external/matpower/case_ieee30.m` | MATPOWER (github.com/MATPOWER/matpower, `data/`), IEEE 30-bus case converted from the IEEE Common Data Format. Topology, reactances and bus loads (283.4 MW) | b3bcc6165577e9e70a39af98f808405233a184289056cab8ce35fbc5a3ad953e |
| `external/matpower/case30.m` | MATPOWER, same 41 branches in the same order. Only its `rateA` column is used, the line limits of Alsac and Stott (1974) | 3d9030311259b553be85d02336b7e1bcb24ec04775bee6671bdb62d18e4e2137 |

Conversion (P in MW): a = c_Abido/1e4, b = b_Abido/100, c = a_Abido, alpha = 1e-2 alpha_Abido,
beta = 1e-2 beta_Abido/100, gamma = 1e-2 gamma_Abido/1e4, zeta = zeta_Abido, lambda = lambda_Abido/100, with emission
in ton/h and cost in $/h. There are no valve-point terms.

Assumptions. Abido gives the generator limits and the bus data only through his references [5], [8], which we could not
obtain. We place G1 to G6 at the standard generator buses 1, 2, 5, 8, 11 and 13 and give every unit limits of 5 and 150 MW.
Without line limits the minimum-cost and minimum-emission dispatches at 283.4 MW reproduce Abido's Table II (Case 1)
within 4e-4 p.u. and his minimum cost of 600.11 $/h (`results/raw/additional/network/abido_check.json`), so the limits do
not bind at these anchors. Our emissions are 0.0010 ton/h above his Table II at both anchors. The offset equals a change
of alpha_4 from the printed 5.426 to 5.326, so his Tables I and II differ in this constant. We keep the printed value.
