# See paper: "Control Variates for Constrained Variables"
# Link: https://ieeexplore.ieee.org/document/9944852
# 
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


# Estimating \alpha using leave one out approach
def linear_control_variates_loo(samples, grad_log_prob):
    try:
        N, dim = samples.shape
        control = -0.5 * grad_log_prob
        zv = []
        for n in range(N):
            samples_loo = np.concatenate([samples[0:n], samples[(n+1):]], axis=0)
            control_loo = np.concatenate([control[0:n], control[(n+1):]], axis=0)
            sc_matrix_loo = np.concatenate((samples_loo, control_loo), axis=1)
            sc_cov_loo = np.cov(sc_matrix_loo.T)
            Sigma_cs_loo = sc_cov_loo[0:dim, dim:dim*2].T
            Sigma_cc_loo = sc_cov_loo[dim:dim*2, dim:dim*2]
            zv_ = -np.linalg.solve(Sigma_cc_loo, Sigma_cs_loo).T @ control[n, :].reshape((dim, 1))
            zv.append(zv_)
        zv = np.concatenate(zv, axis=1)
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
        
        if dim_unconstrained_samples < 50:
            dim_cp = int(0.5*dim_unconstrained_samples*(dim_unconstrained_samples-1))
            dim_control = dim_unconstrained_samples+dim_unconstrained_samples+dim_cp
            z = -0.5 * grad_log_prob
            control = np.concatenate((z, (unconstrained_samples*z - 0.5)), axis=1)
            control_parts = np.zeros((num_samples_total, dim_cp))
            for i in range(2, dim_unconstrained_samples+1):
                for j in range(1, i):
                    ind = int(0.5*(2*dim_unconstrained_samples-j)*(j-1)) + (i-j)
                    control_parts[:,ind-1] = unconstrained_samples[:,i-1]*z[:,j-1] + unconstrained_samples[:,j-1]*z[:,i-1]
            control = np.concatenate((control, control_parts), axis=1)
        else:
            print('WARNING... The dimentionality of the problem is too large ( > 50), using reduced control variates for the quadratic version.')
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

# Estimating \alpha using leave one out approach
def quadratic_control_variates_loo(constrained_samples, unconstrained_samples, grad_log_prob):
    try:
        N = constrained_samples.shape[0]
        dim_constrained_samples = constrained_samples.shape[1]
        dim_unconstrained_samples = unconstrained_samples.shape[1]
        
        if dim_unconstrained_samples < 50:
            dim_cp = int(0.5*dim_unconstrained_samples*(dim_unconstrained_samples-1))
            dim_control = dim_unconstrained_samples+dim_unconstrained_samples+dim_cp
            z = -0.5 * grad_log_prob
            control = np.concatenate((z, (unconstrained_samples*z - 0.5)), axis=1)
            control_parts = np.zeros((N, dim_cp))
            for i in range(2, dim_unconstrained_samples+1):
                for j in range(1, i):
                    ind = int(0.5*(2*dim_unconstrained_samples-j)*(j-1)) + (i-j)
                    control_parts[:,ind-1] = unconstrained_samples[:,i-1]*z[:,j-1] + unconstrained_samples[:,j-1]*z[:,i-1]
            control = np.concatenate((control, control_parts), axis=1)
        else:
            print('WARNING... The dimentionality of the problem is too large ( > 50), using reduced control variates for the quadratic version.')
            dim_control = dim_unconstrained_samples+dim_unconstrained_samples
            z = -0.5 * grad_log_prob
            control = np.concatenate((z, (unconstrained_samples*z - 0.5)), axis=1)
        
        dim_control = control.shape[1]
        zv = []
        for n in range(N):
            samples_loo = np.concatenate([constrained_samples[0:n], constrained_samples[(n+1):]], axis=0)
            control_loo = np.concatenate([control[0:n], control[(n+1):]], axis=0)
            sc_matrix_loo = np.concatenate((samples_loo, control_loo), axis=1)
            sc_cov_loo = np.cov(sc_matrix_loo.T)
            Sigma_cs_loo = sc_cov_loo[0:dim_constrained_samples, dim_constrained_samples:dim_constrained_samples+dim_control].T
            Sigma_cc_loo = sc_cov_loo[dim_constrained_samples:dim_constrained_samples+dim_control, dim_constrained_samples:dim_constrained_samples+dim_control]
            zv_ = -np.linalg.solve(Sigma_cc_loo, Sigma_cs_loo).T @ control[n, :].reshape((dim_control, 1))
            zv.append(zv_)
        zv = np.concatenate(zv, axis=1)

        # sc_matrix = np.concatenate((constrained_samples.T, control.T), axis=0)
        # sc_cov = np.cov(sc_matrix)
        # Sigma_cs = sc_cov[0:dim_constrained_samples, dim_constrained_samples:dim_constrained_samples+dim_control].T
        # Sigma_cc = sc_cov[dim_constrained_samples:dim_constrained_samples+dim_control, dim_constrained_samples:dim_constrained_samples+dim_control]
        # zv = -np.linalg.solve(Sigma_cc, Sigma_cs).T @ control.T

        quad_cv_samples = constrained_samples + zv.T
    except:
        quad_cv_samples = np.empty_like(constrained_samples)
        quad_cv_samples[:] = np.nan

    return quad_cv_samples