# See paper: "Control Variates for Constrained Variables"
# Link: https://ieeexplore.ieee.org/document/9944852
# 
# Note that we have considered a leave-one-out estimator but not found it to make 
# a statistically significant improvement in performance in the contexts we have considered.
# See branch "Leave-one-out-alpha"
import numpy as np


def linear_control_variates(samples, grad_log_prob):
    try:
        dim = samples.shape[1]
        control = -0.5 * grad_log_prob

        sc_matrix = np.concatenate((samples, control), axis=1)
        sc_cov = np.cov(sc_matrix.T)
        Sigma_cs = sc_cov[0:dim, dim:dim*2].T
        Sigma_cc = sc_cov[dim:dim*2, dim:dim*2]
        zv = -np.linalg.solve(Sigma_cc, Sigma_cs).T @ control.T

        linear_cv_samples = samples + zv.T
    except:
        linear_cv_samples = np.empty_like(samples)
        linear_cv_samples[:] = np.nan
    return linear_cv_samples


def quadratic_control_variates(constrained_samples, unconstrained_samples, grad_log_prob):
    try:
        num_samples_total = constrained_samples.shape[0]
        dim_constrained_samples = constrained_samples.shape[1]
        dim_unconstrained_samples = unconstrained_samples.shape[1]
        dim_cp = int(0.5*dim_unconstrained_samples*(dim_unconstrained_samples-1))
        dim_control = dim_unconstrained_samples+dim_unconstrained_samples+dim_cp
        if num_samples_total > dim_control:
            z = -0.5 * grad_log_prob
            control = np.concatenate((z, (unconstrained_samples*z - 0.5)), axis=1)
            control_parts = np.zeros((num_samples_total, dim_cp))
            for i in range(2, dim_unconstrained_samples+1):
                for j in range(1, i):
                    ind = int(0.5*(2*dim_unconstrained_samples-j)*(j-1)) + (i-j)
                    control_parts[:,ind-1] = unconstrained_samples[:,i-1]*z[:,j-1] + unconstrained_samples[:,j-1]*z[:,i-1]
            control = np.concatenate((control, control_parts), axis=1)
        else:
            print('WARNING... The number of samples is smaller than the number of control variates, using reduced control variates for the quadratic version.')
            dim_control = dim_unconstrained_samples+dim_unconstrained_samples
            z = -0.5 * grad_log_prob
            control = np.concatenate((z, (unconstrained_samples*z - 0.5)), axis=1)
        sc_matrix = np.concatenate((constrained_samples.T, control.T), axis=0)
        sc_cov = np.cov(sc_matrix)
        Sigma_cs = sc_cov[0:dim_constrained_samples, dim_constrained_samples:dim_constrained_samples+dim_control].T
        Sigma_cc = sc_cov[dim_constrained_samples:dim_constrained_samples+dim_control, dim_constrained_samples:dim_constrained_samples+dim_control]
        zv = -np.linalg.solve(Sigma_cc, Sigma_cs).T @ control.T

        quad_cv_samples = constrained_samples + zv.T
    except:
        quad_cv_samples = np.empty_like(constrained_samples)
        quad_cv_samples[:] = np.nan

    return quad_cv_samples