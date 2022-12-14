import stan
import json
import bridgestan.python.bridgestan as bs
import numpy as np
from controlvariates.postprocess_bs import run_postprocess, pystan3samples_to_matrix
from bridgestan.python.bridgestan.compile import set_cmdstan_path

# set cmdstan path
set_cmdstan_path('../cmdstan/')

# module path
exp_path = 'stan_benchmark/arma/arma'
# exp_path = 'bridgestan/test_models/logistic/logistic'
model_path = exp_path + '.stan'
data_path = exp_path + '.data.json'

# read stan model
with open(model_path) as model_file:
    model_str = model_file.read()

# Opening JSON file
with open(data_path) as json_file:
    data = json.load(json_file)

posterior = stan.build(model_str, data=data)
fit = posterior.sample(num_chains=3, num_samples=500, num_warmup=500)
#f = fit.to_frame()  # pandas `DataFrame, requires pandas

# initialise BridgeStan
model = bs.StanModel.from_stan_file(model_path, data_path)

# extract samples from pystan3
samples = {}
for name in fit.param_names:
    samples[name] = fit[name]
constrained_samples, name_parameters = pystan3samples_to_matrix(samples, fit.num_chains*fit.num_samples, model)

# post-process using control variates
cv_samples, times = run_postprocess(constrained_samples, model, cv_mode='linear', output_squared_samples=False, output_runtime=True)
cv_samples_quadratic, times_quadratic = run_postprocess(constrained_samples, model, cv_mode='quadratic', output_squared_samples=False, output_runtime=True)

# evaluate using a very large number of samples
fit_larger = posterior.sample(num_chains=3, num_samples=10000, num_warmup=10000)
samples_larger = {}
for name in fit_larger.param_names:
    samples_larger[name] = fit_larger[name]
constrained_samples_larger, name_parameters = pystan3samples_to_matrix(samples_larger, fit_larger.num_chains*fit_larger.num_samples, model)

raw_samples_mean = np.mean(constrained_samples, axis=0)
cv_linear_mean = np.mean(cv_samples, axis=0)
cv_quad_mean = np.mean(cv_samples_quadratic, axis=0)
larger_samples_mean = np.mean(constrained_samples_larger, axis=0)
raw_rmse = np.sqrt(np.mean((raw_samples_mean - larger_samples_mean)**2))
cv_linear_rmse = np.sqrt(np.mean((cv_linear_mean - larger_samples_mean)**2))
cv_quad_rmse = np.sqrt(np.mean((cv_quad_mean - larger_samples_mean)**2))
print('means rmse: {:f} -- {:f} -- {:f}'.format(raw_rmse, cv_linear_rmse, cv_quad_rmse))

raw_samples_var = np.var(constrained_samples, axis=0)
print('raw sample variances:')
print(raw_samples_var)
cv_linear_var = np.var(cv_samples, axis=0)
print('control variates (linear) sample variances:')
print(cv_linear_var)
cv_quad_var = np.var(cv_samples_quadratic, axis=0)
print('control variates (quadratic) sample variances:')
print(cv_quad_var)
larger_samples_var = np.var(constrained_samples_larger, axis=0)
print('larger raw sample (as groundtruth) variances:')
print(larger_samples_var)
print('- - - - - - - - - - - - - - - - - - - - - - ')
