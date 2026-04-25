# BYSpec - BFOSC and YFOCS Spectral Data Reduction Pipeline

BYSpec is an open‑source, fully automatic data reduction pipeline
for long‑slit and echelle spectroscopic data obtained with the
BFOSC (Xinglong 2.16m telescope) and YFOSC (Lijiang 2.4m telescope)
instruments.

The package handles all standard reduction steps with minimal user intervention:

* Bias and overscan correction
* Spectroscopic flat‑fielding (tailored for both long‑slit and echelle modes)
* Spectral curvature ("smile") correction for long‑slit data
* Automatic order detection and background subtraction
* Wavelength calibration using built‑in line lists (FeAr, HeNe, etc.)
  and pre‑computed templates
* Optimal extraction to maximize S/N and remove cosmic rays

