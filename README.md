# BO-HBM-ex
[![GitHub license](https://img.shields.io/github/license/ragq/BO-HBM-ex)](https://github.com/ragq/BO-HBM-ex) [![GitHub release](https://img.shields.io/github/release/ragq/BO-HBM-ex.svg)](https://github.com/ragq/BO-HBM-ex/releases/) [![GitHub stars](https://img.shields.io/github/stars/ragq/BO-HBM-ex)](https://github.com/ragq/BO-HBM-ex/stargazers) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10259290.svg)](https://doi.org/10.5281/zenodo.10259290)

[![Binder - Parametric](https://img.shields.io/badge/launch%20Binder-Parametric%20study%20&%20Optimization-579ACA.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFkAAABZCAMAAABi1XidAAAB8lBMVEX///9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olJXmsrmZYH1olL1olL0nFf1olJXmsrmZYH1olJXmsq8dZb1olJXmsrmZYH1olJXmspXmspXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olLeaIVXmsrmZYH1olL1olL1olJXmsrmZYH1olLna31Xmsr1olJXmsr1olJXmsrmZYH1olLqoVr1olJXmsr1olJXmsrmZYH1olL1olKkfaPobXvviGabgadXmsqThKuofKHmZ4Dobnr1olJXmsr1olJXmspXmsr1olJXmsrfZ4TuhWn1olL1olJXmsqBi7X1olJXmspZmslbmMhbmsdemsVfl8ZgmsNim8Jpk8F0m7R4m7F5nLB6jbh7jbiDirOEibOGnKaMhq+PnaCVg6qWg6qegKaff6WhnpKofKGtnomxeZy3noG6dZi+n3vCcpPDcpPGn3bLb4/Mb47UbIrVa4rYoGjdaIbeaIXhoWHmZYHobXvpcHjqdHXreHLroVrsfG/uhGnuh2bwj2Hxk17yl1vzmljzm1j0nlX1olL3AJXWAAAAbXRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hgYGBkcHBwcXl8gICAgoiIkJCQlJicnJ2goKCmqK+wsLC4usDAwMjP0NDQ1NbW3Nzg4ODi5+3v8PDw8/T09PX29vb39/f5+fr7+/z8/Pz9/v7+zczCxgAABC5JREFUeAHN1ul3k0UUBvCb1CTVpmpaitAGSLSpSuKCLWpbTKNJFGlcSMAFF63iUmRccNG6gLbuxkXU66JAUef/9LSpmXnyLr3T5AO/rzl5zj137p136BISy44fKJXuGN/d19PUfYeO67Znqtf2KH33Id1psXoFdW30sPZ1sMvs2D060AHqws4FHeJojLZqnw53cmfvg+XR8mC0OEjuxrXEkX5ydeVJLVIlV0e10PXk5k7dYeHu7Cj1j+49uKg7uLU61tGLw1lq27ugQYlclHC4bgv7VQ+TAyj5Zc/UjsPvs1sd5cWryWObtvWT2EPa4rtnWW3JkpjggEpbOsPr7F7EyNewtpBIslA7p43HCsnwooXTEc3UmPmCNn5lrqTJxy6nRmcavGZVt/3Da2pD5NHvsOHJCrdc1G2r3DITpU7yic7w/7Rxnjc0kt5GC4djiv2Sz3Fb2iEZg41/ddsFDoyuYrIkmFehz0HR2thPgQqMyQYb2OtB0WxsZ3BeG3+wpRb1vzl2UYBog8FfGhttFKjtAclnZYrRo9ryG9uG/FZQU4AEg8ZE9LjGMzTmqKXPLnlWVnIlQQTvxJf8ip7VgjZjyVPrjw1te5otM7RmP7xm+sK2Gv9I8Gi++BRbEkR9EBw8zRUcKxwp73xkaLiqQb+kGduJTNHG72zcW9LoJgqQxpP3/Tj//c3yB0tqzaml05/+orHLksVO+95kX7/7qgJvnjlrfr2Ggsyx0eoy9uPzN5SPd86aXggOsEKW2Prz7du3VID3/tzs/sSRs2w7ovVHKtjrX2pd7ZMlTxAYfBAL9jiDwfLkq55Tm7ifhMlTGPyCAs7RFRhn47JnlcB9RM5T97ASuZXIcVNuUDIndpDbdsfrqsOppeXl5Y+XVKdjFCTh+zGaVuj0d9zy05PPK3QzBamxdwtTCrzyg/2Rvf2EstUjordGwa/kx9mSJLr8mLLtCW8HHGJc2R5hS219IiF6PnTusOqcMl57gm0Z8kanKMAQg0qSyuZfn7zItsbGyO9QlnxY0eCuD1XL2ys/MsrQhltE7Ug0uFOzufJFE2PxBo/YAx8XPPdDwWN0MrDRYIZF0mSMKCNHgaIVFoBbNoLJ7tEQDKxGF0kcLQimojCZopv0OkNOyWCCg9XMVAi7ARJzQdM2QUh0gmBozjc3Skg6dSBRqDGYSUOu66Zg+I2fNZs/M3/f/Grl/XnyF1Gw3VKCez0PN5IUfFLqvgUN4C0qNqYs5YhPL+aVZYDE4IpUk57oSFnJm4FyCqqOE0jhY2SMyLFoo56zyo6becOS5UVDdj7Vih0zp+tcMhwRpBeLyqtIjlJKAIZSbI8SGSF3k0pA3mR5tHuwPFoa7N7reoq2bqCsAk1HqCu5uvI1n6JuRXI+S1Mco54YmYTwcn6Aeic+kssXi8XpXC4V3t7/ADuTNKaQJdScAAAAAElFTkSuQmCC)](https://mybinder.org/v2/gh/RagQ/BO-HBM-ex/dev?urlpath=%2Fdoc%2Ftree%2Fhome.ipynb)


## Description

The Python's tool `BO-HBM-ex` shows an example of the application of Bayesian Optimization [1] on a Duffing problem [2]. The Duffing's oscillator problem is here solved using Harmonic Balance Method and a continuation technique.

<img src="illus/duffing_scheme.png" width="200" ><br>
Duffing oscillator scheme. 

The optimization problem can be written:

------------

Find $`(k_{nl}^*,\xi^*)`$ such as 

$`(k_{nl}^*,\xi^*)=\underset{(k_{nl},\xi)\in\mathcal{D}}{\arg\min}\,\underset{\omega\in[\omega_l,\omega_u]}{\max} \ddot{q}_{\mathrm{RMS}}(k_{nl},\xi,\omega)`$

-------------


## Installation

Get the source code from this repository. The example can be run after installation of dependencies with 

```pip install -r requirements.txt```


## Usage

The Python's scripts can be run only with Python 3.

### Run solver of Duffing oscillator
The file `respDuffing.py` provides frequencies responses for displacement and acceleration on PDF pictures for $`\xi=\{0.15, 0.3, 0.5, 1.\}`$ ($[\mathrm{kg}\cdot\mathrm{s}^{-1}]$) and $`k_{nl}=\{0.25,0.5,0.75,...,2\}`$ ($[\mathrm{N}\cdot\mathrm{m}^{-3}]$) . Pictures are stored in `ParamDuffing` folder.

Following pictures show $\ddot{q}_ {\mathrm{RMS}}$ for a few values of $k_ {nl}$ for $`\xi=0.15\,\mathrm{kg}\cdot\mathrm{s}^{-1}`$ (a), $`\xi=0.3\,\mathrm{kg}\cdot\mathrm{s}^{-1}`$ (b), $`\xi=0.5\,\mathrm{kg}\cdot\mathrm{s}^{-1}`$ (c) and  $`\xi=1\,\mathrm{kg}\cdot\mathrm{s}^{-1}`$ (d).

(a)|(b)
:---:|:---:
![Arms_xi15](illus/Arms_xi15.png) | ![Arms_xi30](illus/Arms_xi30.png)
(c)|(d)
![Arms_xi50](illus/Arms_xi50.png) |  ![Arms_xi100](/illus/Arms_xi100.png)

Animations of $q_ {\mathrm{RMS}}$ (a) and $\ddot{q}_ {\mathrm{RMS}}$  (b) of large set of values of $k_{nl}$ for $`\xi=0.15\,\mathrm{kg}\cdot\mathrm{s}^{-1}`$:

<!-- _{\mathrm{RMS}}$ -->

(a)|(b)
:---:|:---:
![anim_Drms](/illus/anim_Drms-optim.gif) |  ![anim_Arms](/illus/anim_Arms-optim.gif)

### Run Bayesian Optimization on Duffing oscillator

The file `OptiExp.py` generates data of Bayesian Optimization's iterations. These data will be available on the directory `ExpOptimDuffing` which contains data for sample sets containing 10, 20 and 25 samples. Results are provided along BO iterations on `CSV` files and acquisition and objective functions are plotted in 2D and 3D.

The following pictures show the evolution of the acquisition and objective functions along BO's iterations applied on the Optimization of the Duffing Oscillator (minimization of the maximum of the acceleration along the frequency bandwidth $[0,2.5]$ ($\mathrm{rad}\cdot\mathrm{s}^{-1}$)). The initial sampling obtained with LHS contains 10 sample points.

Acquisition function               |  Objective function
:---:|:---:
![anim_10_contourEI](/illus/anim_10_contourEI-optim.gif) |  ![anim_10_contourObj](/illus/anim_10_contourObj-optim.gif)
![anim_10_surfaceEI](/illus/anim_10_surfaceEI-optim.gif) |  ![anim_10_surfaceObj](/illus/anim_10_surfaceObj-optim.gif)


### Versions

The code has been executed without any issues with the following versions of Python and libraries:
``````
- Python 3.10.9
- numpy 1.26.2
- matplotlib 3.8.2
- pandas 2.1.3
- torch 2.1.1
- botorch 0.9.4
- gpytorch 1.11
- pydoe 0.3.8
``````

## How to cite

This repo is relative to the [PhD thesis](https://www.theses.fr/s263751) of [Quentin Ragueneau](https://www.lmssc.cnam.fr/fr/user/209) achieved at [LMSSC](https://www.lmssc.cnam.fr) under the supervision of [Antoine Legay](https://www.lmssc.cnam.fr/fr/equipe/permanents/antoine-legay) and [Luc Laurent](https://www.lmssc.cnam.fr/fr/equipe/luc-laurent) in collaboration with [Ingeliance Technologies](https://www.ingeliance.com) and founded by [ANRT](https://www.anrt.asso.fr/fr) (Cifre 2020/0272).

Please use the following citation reference if you use the code:

`Ragueneau, Q., Laurent, L. & Legay, A. (2023). BO-HBM-ex (vxxx). Zenodo. https://doi.org/10.5281/zenodo.102592910`

Bibtex entry:
``````
@software{BO-HBM-ex-soft,
author       = {Ragueneau, Quentin and Laurent, Luc and Legay, Antoine},
title        = {{BO-HBM-ex}},
month        = dec,
year         = 2023,
publisher    = {Zenodo},
version      = {vxxx}
doi          = {10.5281/zenodo.10259290},
url          = {https://doi.org/10.5281/zenodo.10259290}
}
``````
NB: version number and DOI must be adapted from [Zenodo's repository](https://doi.org/10.5281/zenodo.10259290).

## License

MIT License

Copyright (c) 2023 - Quentin Ragueneau (quentin.ragueneau@ingeliance.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## References
```
[1] D. R. Jones, M. Schonlau, and W. J. Welch. Efficient Global Optimization of Expensive Black-Box Functions. Journal of Global Optimization, 13(4):455–492, Dec. 1998.

[2] B. Balaram, M. D. Narayanan, and P. K. Rajendrakumar. Optimal design of multi-parametric nonlinear systems using a parametric continuation based Genetic Algorithm approach. Nonlinear Dynamics, 67(4):2759–2777, Mar. 2012.
```
