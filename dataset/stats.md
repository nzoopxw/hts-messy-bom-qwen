# Messy-BOM HTS dataset — build stats

- Candidate rows read: 4025
- Rows with a 10-digit label: 2981
- After (text, code) dedup: 2603
- Eval rows dropped for part-number leakage into train: 53
- Eval rows dropped for conflicting labels or part-number-only text: 18
- Projects: 91 (train 50, val 15, test 26)

| split | rows | real | augmented | extra | distinct codes | headings (4-digit) | gold labels | copied from distributor | codes unseen in train |
|---|---|---|---|---|---|---|---|---|---|
| train | 2095 | 1176 | 919 | 0 | 108 | 29 | 6 | 8 |  |
| val | 164 | 164 | 0 | 0 | 38 | 9 | 0 | 12 | 3 rows (0 with unseen heading) |
| test | 237 | 237 | 0 | 0 | 53 | 16 | 1 | 2 | 6 rows (1 with unseen heading) |

## Most common headings (real rows, all splits)

| heading | rows |
|---|---|
| 8541 | 384 |
| 8542 | 314 |
| 8536 | 241 |
| 8533 | 231 |
| 8532 | 208 |
| 8504 | 82 |
| 8548 | 40 |
| 8538 | 13 |
| 7318 | 9 |
| 8517 | 9 |
| 8544 | 9 |
| 8543 | 6 |
| 8471 | 5 |
| 7326 | 4 |
| 8529 | 3 |
| 8518 | 2 |
| 8473 | 2 |
| 3926 | 2 |
| 8531 | 2 |
| 8477 | 1 |

## Example test rows

| text | hts |
|---|---|
| DS1307Z+T&R Real Time Clock IC SOIC-8 | 8542.39.0090 |
| 1n 0603 | 8532.24.0020 |
| MM3Z16VT1 16V 6% 0.2W Zener Diode SOD-323 | 8541.10.0050 |
| 6.8pF C_0402 | 8532.24.0020 |
| C_100n_0402 50V X5R C_0402 | 8532.24.0020 |
| MMBT3906 45V Vce, 0.8A Ic, PNP, SOT-23 SOT-23 | 8541.21.0075 |
| R_10k2_0402 1% R_0402 | 8533.21.0030 |
| FB_600Z_0.5A_0603 FB_0603 | 8548.00.0000 |
| BAV99 SOT-23 | 8541.10.0070 |
| pin header, 1x08, 2.54mm pitch, single row | 8536.69.4040 |
| 100nF C_Small: Unpolarized capacitor, small symbol 0805 | 8532.24.0020 |
| C 2.2uF Unpolarized capacitor 0603 1608Metric | 8532.24.0020 |
