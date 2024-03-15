# Fast Screened Poisson LGF <a href="https://github.com/WeiHou1996/Fast-Screened-Poisson-LGF/blob/main/LICENSE.md"> <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
This repository provides a set of algorithms to efficiently evaluate the lattice Green's function (LGF) of the screened Poisson equation. This LGF is the fundamental solution of the discrete screened Poisson equation. It finds its application in many areas such as:

  1. Solving the incompressible Navier-Stokes equations
  2. Computing the return probabilities of a random walk with killing
  3. Modeling the effects of material impurities
  4. Many others...

The file "src/LGF_funcs.py" contains many methods to tabulate the LGF of the screened Poisson equation. The files in the folder "Comparisons" compare the runtime and accuracy of various methods. The files in the folder "Applications" demonstrate the potential use of the LGF of the screened Poisson equation. The scripts described above are used to generate the performance results of the following paper using an Apple M1 Pro chip.

```bibtex
@article{hou2024lattice,   
     title={Fast and robust method for screened {P}oisson lattice {G}reen's function using asymptotic expansion and {F}ast {F}ourier {T}ransform},   
     author={Hou, Wei and Colonius, Tim},   
     journal={arXiv preprint arXiv:2403.03076},   
     year={2024}   
 }   
 ```

## Reference
[1] Kotera, Takeyasu. "Localized vibration and random walk." Progress of Theoretical Physics Supplement 23 (1962): 141-156.

[2] Katsura, Shigetoshi, and Sakari Inawashiro. "Lattice Green's functions for the rectangular and the square lattices at arbitrary points." Journal of Mathematical Physics 12.8 (1971): 1622-1630.

[3] Martinsson, Per-Gunnar, and Gregory J. Rodin. "Asymptotic expansions of lattice Green's functions." Proceedings of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences 458.2027 (2002): 2609-2622.

[4] Lawler, Gregory F., and Vlada Limic. Random walk: a modern introduction. Vol. 123. Cambridge University Press, 2010.

[5] Gabbard, James, and Wim M. van Rees. "Lattice Green’s Functions for High-Order Finite Difference Stencils." SIAM Journal on Numerical Analysis 62.1 (2024): 25-47.

[6] Hou, Wei, and Tim Colonius. "An adaptive lattice Green's function method for external flows with two unbounded and one homogeneous directions." arXiv preprint arXiv:2402.13370 (2024).

## License
 
Copyright 2024.
This code is under the MIT license (see [LICENSE](https://github.com/WeiHou1996/Fast-Screened-Poisson-LGF/blob/main/LICENSE) file for full text).
