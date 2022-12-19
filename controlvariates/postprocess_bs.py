import copy
import time
import numpy as np
from controlvariates.controlvariates_basics import linear_control_variates, quadratic_control_variates


def pystan3samples_to_matrix(samples, num_samples, model_bs):
    # arrange constrained samples
    samples_flatten = {}
    for name in samples.keys():
        samples_per_param = np.moveaxis(samples[name], -1, 0)
        samples_flatten_per_param = []
        for i in range(num_samples):
            flatten_ = samples_per_param[i].flatten()
            samples_flatten_per_param.append(flatten_.tolist())
        samples_flatten[name] = np.array(samples_flatten_per_param)

    name_parameters = model_bs.param_names(include_tp=False, include_gq=False)
    constrained_samples = []
    for name in name_parameters:
        dot_position = name.find('.')
        if dot_position >= 0:
            name_prefix = name[0:dot_position]
            ind = int(name[dot_position+1:])-1
            constrained_samples.append(samples_flatten[name_prefix][:, ind])
        else:
            constrained_samples.append(samples_flatten[name][:, 0])
    constrained_samples = np.array(constrained_samples)
    constrained_samples = constrained_samples.T

    return constrained_samples, name_parameters


def run_postprocess(samples, model_bs, cv_mode='linear', output_squared_samples=False, output_runtime=False):
    num_samples = samples.shape[0]

    # Unconstraint mcmc samples.
    unconstrained_samples = []
    for i in range(num_samples):
        unconstrained_samples.append(model_bs.param_unconstrain(copy.copy(samples[i])))
    unconstrained_samples = np.array(unconstrained_samples)

    # Calculate gradients of the log-probability
    grad_start_time = time.time()
    grad_log_prob_vals = []
    for i in range(num_samples):
        log_p, grad = model_bs.log_density_gradient(copy.copy(unconstrained_samples[i]), propto=True, jacobian=True)
        grad_log_prob_vals.append(grad)
    grad_log_prob_vals = np.array(grad_log_prob_vals)
    grad_runtime = time.time() - grad_start_time

    # Run control variates
    cv_start_time = time.time()
    if cv_mode == 'linear':
        cv_samples = linear_control_variates(samples, grad_log_prob_vals)
        cv_runtime = time.time() - cv_start_time
        # print('Gradient time: {:.05f} --- Linear control variate time: {:.05f}.'.format(grad_runtime, cv_runtime))
    elif cv_mode == 'quadratic':
        cv_samples = quadratic_control_variates(samples, unconstrained_samples, grad_log_prob_vals)
        cv_runtime = time.time() - cv_start_time
        # print('Gradient time: {:.05f} --- Quadratic control variate time: {:.05f}.'.format(grad_runtime, cv_runtime))
    else:
        print('The mode of control variates must be linear or quadratic.')
        return None

    if output_squared_samples == True:
        # the squared samples are used for calculating the standard deviation of the problem.
        if cv_mode == 'linear':
            cv_samples_suqared = linear_control_variates(samples**2, grad_log_prob_vals)
        elif cv_mode == 'quadratic':
            cv_samples_suqared = quadratic_control_variates(samples**2, unconstrained_samples, grad_log_prob_vals)
        if output_runtime:
            return cv_samples, cv_samples_suqared, (grad_runtime, cv_runtime)
        else:
            return cv_samples, cv_samples_suqared
    else:
        if output_runtime:
            return cv_samples, (grad_runtime, cv_runtime)
        else:
            return cv_samples
    
