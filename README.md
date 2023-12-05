SQUID repository for manuscript analysis
========================================================================
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10047748.svg)](https://doi.org/10.5281/zenodo.10047748)

<br/>

![logo_dark](./images/logo_dark.png#gh-dark-mode-only)
![logo_light](./images/logo_light.png#gh-light-mode-only)

<br/>

This repository reproduces the analysis performed in our manuscript **Identifying *cis*-regulatory mechanisms from genomic deep neural networks using surrogate models** (Seitz, McCandlish, Kinney and Koo). It contains tools to apply the discussed method **SQUID** (**S**urrogate **Qu**antitative **I**nterpretability for **D**eepnets) on genomic models.

## Usage:
The `/examples` directory contains example code for running SQUID on several deep-learning models. Within that directory, we have supplied additional README material for setting up environments and initializing our workflow. As well, detailed instructions and comments for all procedures are provided in the code.

The release of this repository is also available via [Zenodo](doi.org/10.5281/zenodo.10047747), with the addition that all intermediate and final outputs are provided there.

## Attribution:
If the specific code in this analysis repository is useful in your work, please cite:

```bibtex
@article{seitz2023_squid-manuscript,
  title={SQUID manuscript workflow with outputs},
  author={Seitz, Evan},
  doi={10.5281/zenodo.10047747},
  year={2023},
  publisher={Zenodo}
}
```

For more general usage pertaining to the SQUID method, please see instructions in our frontend SQUID repository:
https://github.com/evanseitz/squid-nn

## License:
Copyright (C) 2022–2023 Evan Seitz, David McCandlish, Peter Koo, Justin Kinney

The software, code sample and their documentation made available on this website could include technical or other mistakes, inaccuracies or typographical errors. We may make changes to the software or documentation made available on its web site at any time without prior notice. We assume no responsibility for errors or omissions in the software or documentation available from its web site. For further details, please see the LICENSE file.
