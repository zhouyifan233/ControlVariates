"""
Stan summary:
mean: sample means
se_mean: standard error for the mean = sd / sqrt(n_eff)
sd: sample standard deviations
quantiles:
n_eff: effective sample size
Rhat:
"""

import numpy as np
import copy
from pystan.misc import _array_to_table


def get_cv_sample_mean_std(fit, cv_samples):
    sample_mean = {}
    sample_std = {}
    means = np.mean(cv_samples, axis=0)
    stds = np.std(cv_samples, axis=0)
    fnames = copy.copy(fit.sim['fnames_oi'])
    fnames.remove('lp__')    # not a parameter in the stan model
    assert(len(fnames) == means.shape[0])
    assert(len(fnames) == stds.shape[0])
    for i, fname in enumerate(fnames):
        sample_mean[fname] = means[i]
        sample_std[fname] = stds[i]
    return sample_mean, sample_std

def get_parameter_std(fit, cv_samples, cv_samples_squared):
    parameter_std = {}
    expect_x = np.mean(cv_samples, axis=0)
    expect_x_squared = np.mean(cv_samples_squared, axis=0)
    fnames = copy.copy(fit.sim['fnames_oi'])
    fnames.remove('lp__')    # not a parameter in the stan model
    assert(len(fnames) == expect_x.shape[0])
    assert(len(fnames) == expect_x_squared.shape[0])
    for i, fname in enumerate(fnames):
        parameter_std[fname] = np.sqrt(expect_x_squared[i] - expect_x[i]**2)
    return parameter_std

def get_cv_sample_ess(fit, cv_samples):
    sim_copy = copy.deepcopy(fit.sim)
    num_save = sim_copy['n_save']
    num_warmups = sim_copy['warmup2']
    num_samples = []
    num_samples_total = 0
    for ns, nw in zip(num_save, num_warmups):
        num_samples.append(ns - nw)
        num_samples_total += ns - nw
    fnames = copy.copy(sim_copy['fnames_oi'])
    fnames.remove('lp__')    # not a parameter in the stan model
    assert(len(fnames) == cv_samples.shape[1])
    # reuse Pystan ess function to calculate ess of control variates samples
    # substitute the MCMC samples in fit by control variate samples
    for ci, chain in enumerate(sim_copy['samples']):
        for fi, fname in enumerate(fnames):
            chain['chains'][fname][num_warmups[ci]:num_save[ci]] = cv_samples[ci*num_samples[ci]:(ci+1)*num_samples[ci], fi]
    ess = {}
    for fi, fname in enumerate(fnames):
        ess[fname] = pystan.chains.ess(sim_copy, fi)
    return ess

def get_sample_ess(fit):
    # Call stan built-in function to calculate ESS.
    sim_copy = copy.deepcopy(fit.sim)
    fnames = copy.copy(sim_copy['fnames_oi'])
    fnames.remove('lp__')    # not a parameter in the stan model
    ess = {}
    for fi, fname in enumerate(fnames):
        ess[fname] = pystan.chains.ess(sim_copy, fi)
    return ess

def get_se_mean(sd, ess):
    fnames = list(sd.keys())
    se_means = {}
    for fname in fnames:
        se_means[fname] = sd[fname] / np.sqrt(ess[fname])
    return se_means

def stansummary_control_variates(fit, cv_samples, cv_samples_squared, digits_summary=2):
    parameters_mean, cv_sample_std = get_cv_sample_mean_std(fit, cv_samples)
    parameters_std = get_parameter_std(fit, cv_samples, cv_samples_squared)
    cv_sample_ess = get_cv_sample_ess(fit, cv_samples)
    mcmc_sample_ess = get_sample_ess(fit)
    se_mean = get_se_mean(cv_sample_std, cv_sample_ess)

    fnames = list(parameters_mean.keys())
    content = np.array([list(parameters_mean.values()), list(se_mean.values()), list(parameters_std.values()), list(cv_sample_ess.values())])
    body = _array_to_table(content.T, fnames, ['mean', 'se_mean', 'sd', 'n_eff'], digits_summary)

    return body
