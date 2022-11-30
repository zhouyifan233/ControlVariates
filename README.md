# ControlVariates
Postprocess MCMC samples using control variates to get samples with small variance.

Bridgestan (python version) is used to expose necessary parameters. 

# Requirement

Installed and compiled [CmdStan](https://github.com/stan-dev/cmdstan)

The example uses pystan3 to generate MCMC samples:

```shell
$ pip install pystan
```

# Install

```shell
$ git clone https://github.com/zhouyifan233/ControlVariates --recursive
```

# Run example

Configure cmdstan path in example_cv.py (line 9).

```shell
$ cd ControlVariates
$ python example_cv.py
```

# Paper
https://ieeexplore.ieee.org/document/9944852

Please consider cite our work if you find it useful:

@ARTICLE{9944852,
  author={Maskell, Simon and Zhou, Yifan and Mira, Antonietta},
  journal={IEEE Signal Processing Letters}, 
  title={Control Variates for Constrained Variables}, 
  year={2022},
  volume={29},
  number={},
  pages={2333-2337},
  doi={10.1109/LSP.2022.3221347}}



