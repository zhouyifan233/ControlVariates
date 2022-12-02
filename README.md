# ControlVariates
Still under construction, but should be able to use. Any comments are welcome.


"Control variates" is a postprocess technique to reduce the variance of MCMC samples. It can achieve better approximation of expectation of random variables with small number of samples, see the example in example_cv.py.

The processing chain is:

```
1. Draw samples using any tool. (e.g. pystan3)

2. Generate a matrix including the samples. (e.g. pystan3samples_to_matrix() in postprocess_bs.py)

3. Extract gradient of log-probability. (e.g. using bridgestan to expose (see run_postprocess() in postprocess_bs.py))

4. Construct control variates and generate new samples. (see controlvariates_basics.py)
```

# Requirement

- Installed and compiled [CmdStan](https://github.com/stan-dev/cmdstan)

- The example uses pystan3 to generate MCMC samples:

```shell
$ pip install pystan
```

- Any dependants that are required by cmdstan and pystan3.

# Install

```shell
$ git clone https://github.com/zhouyifan233/ControlVariates --recursive
```

Bridgestan will be automatically downloaded.

# Run example

**Configure cmdstan path in example_cv.py (line 9).**

```shell
$ cd ControlVariates
$ python example_cv.py
```

You can try different models: just change exp_path (line 12).

# Paper
[Control Variates for Constrained Variables](https://ieeexplore.ieee.org/document/9944852)


Please consider cite our work if you find it useful:

@ARTICLE{9944852,

  author={Maskell, Simon and Zhou, Yifan and Mira, Antonietta},

  journal={IEEE Signal Processing Letters}, 

  title={Control Variates for Constrained Variables}, 

  year={2022},

  volume={29},

  number={},

  pages={2333-2337},

  doi={10.1109/LSP.2022.3221347}
  
  }



