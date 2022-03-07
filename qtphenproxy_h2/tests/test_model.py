import numpy as np
import pytest
import statsmodels.api as sm
import torch

import qtphenproxy_h2.model as model


def sigmoid(data):
    return 1 / (np.exp(-data) + 1)


def generate_dataset(family='gaussian', N=1000, n_features=5, intercept=True, seed=0):
    np.random.seed(seed)
    true_betas = np.random.normal(size=n_features)
    true_intercept = np.zeros(1) if not intercept else np.random.normal(size=1)
    true_parameters = np.concatenate([true_intercept, true_betas])

    X = np.random.normal(size=N * n_features).reshape((N, n_features))
    X = sm.add_constant(X, prepend=True)
    l = X @ true_parameters

    family_to_data_function = {
        'gaussian': lambda x: x + np.random.normal(size=N, scale=0.1),
        'binomial': lambda x: np.random.binomial(1, p=sigmoid(x), size=N),
        'poisson': lambda x: np.random.poisson(np.exp(x), size=N)
    }
    data_function = family_to_data_function[family]
    y = data_function(l)
    return {'X': X, 'y': y, 'beta': true_betas, 'intercept': true_intercept,
            'params': np.concatenate([true_intercept, true_betas])}


@pytest.mark.parametrize("family", ['gaussian', 'binomial', 'poisson'])
@pytest.mark.parametrize("seed", range(1))
def test_generate_data(family, seed):
    """Check that generated data are consistent"""
    data_1 = generate_dataset(family=family, seed=seed)
    data_2 = generate_dataset(family=family, seed=seed)
    for key, value in data_1.items():
        assert value == pytest.approx(data_2[key])


@pytest.mark.parametrize("family", ['gaussian', 'binomial', 'poisson'])
@pytest.mark.parametrize("N", [100_000])
@pytest.mark.parametrize("n_features", [1, 2, 5])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("seed", range(2))
def test_phenotype_fit(family, seed, n_features, intercept, N, error_weight=1, heritability_weight=0, l1_weight=0,
                       l2_weight=0, learning_rate=0.01, n_iter=2_500):
    """
    Test that qtphenproxy_h2.model.PhenotypeFit is able to fit a linear regression properly.
    """
    data = generate_dataset(family=family, N=N, n_features=n_features, intercept=intercept, seed=seed)

    # Check that a statsmodels regression correctly estimates the coefficients
    np.random.seed(seed)
    family_to_sm_family = {
        'gaussian': sm.families.Gaussian(),
        'binomial': sm.families.Binomial(),
        'poisson': sm.families.Poisson()
    }
    sm_mod = sm.GLM(data['y'], data['X'], family=family_to_sm_family[family])
    res = sm_mod.fit()
    sm_params = res.params.flatten()
    sm_preds = res.fittedvalues
    assert sm_params == pytest.approx(data['params'], rel=0.1, abs=0.1)

    # Check that qtphenproxy_h2.model.PhenotypeFit correctly estimates the coefficients
    # Remove the intercept since pytorch handles it differently
    X = torch.from_numpy(data['X'][:, 1:]).float().view(N, n_features)
    y = torch.from_numpy(data['y']).float().view(N, 1)
    torch_mod = model.PhenotypeFit(input_dim=n_features, output_dim=1, family=family, error_weight=error_weight,
                                   heritability_weight=heritability_weight, l1_weight=l1_weight, l2_weight=l2_weight)
    torch_mod.fit(X, y, n_iter=n_iter, learning_rate=learning_rate, seed=seed, verbose=False)
    torch_params = np.concatenate([torch_mod.linear.bias.detach().numpy(),
                                   torch_mod.linear.weight.detach().numpy().flatten()])
    torch_preds = torch_mod(X).detach().numpy().flatten()
    assert torch_params == pytest.approx(data['params'], rel=0.1, abs=0.1)

    # Check that the actual predictions (fitted values) are nearly equal
    assert sm_preds == pytest.approx(torch_preds, rel=0.25, abs=0.25)


@pytest.mark.parametrize("family", ['gaussian', 'binomial', 'poisson'])
@pytest.mark.parametrize("N", [100_000])
@pytest.mark.parametrize("n_features", [1, 2, 5])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("seed", range(2))
def test_multifitter_basic(family, N, n_features, intercept, seed, error_weight=1, heritability_weight=0, l1_weight=0,
                           l2_weight=0, learning_rate=0.01, n_iter=2_500):
    """Check that a multifitter functions exactly like the lower level models"""
    data = generate_dataset(family=family, N=N, n_features=n_features, intercept=intercept, seed=seed)
    X = torch.from_numpy(data['X'][:, 1:]).float().view(N, n_features)
    y = torch.from_numpy(data['y']).float().view(N, 1)

    # Phenotype fitter (lower level model)
    phenotype_mod = model.PhenotypeFit(input_dim=n_features, output_dim=1, family=family, error_weight=error_weight,
                                       heritability_weight=heritability_weight, l1_weight=l1_weight,
                                       l2_weight=l2_weight)
    phenotype_mod.fit(X, y, n_iter=n_iter, learning_rate=learning_rate, seed=seed, verbose=False)
    phenotype_mod_params = np.concatenate([phenotype_mod.linear.bias.detach().numpy(),
                                           phenotype_mod.linear.weight.detach().numpy().flatten()])
    phenotype_mod_preds = phenotype_mod(X).detach().numpy().flatten()
    assert phenotype_mod_params == pytest.approx(data['params'], rel=0.1, abs=0.1)

    # Multifitter (higher level model)
    multi_mod = model.MultiFitter(X=X, y=y, family=family)
    multi_mod.fit_single(error_weight=error_weight, heritability_weight=heritability_weight, l1_weight=l1_weight,
                         l2_weight=l2_weight, seed=seed, learning_rate=learning_rate, n_iter=n_iter)
    multi_mod_params = (
        multi_mod
        .parameters_df
        .loc[(error_weight, heritability_weight, l1_weight, l2_weight, seed, learning_rate, n_iter),
             ['intercept', *tuple(range(n_features))]]
        .values
    )
    multi_mod_preds = multi_mod.get_predictions(error_weight=error_weight, heritability_weight=heritability_weight,
                                                l1_weight=l1_weight, l2_weight=l2_weight, seed=seed,
                                                learning_rate=learning_rate, n_iter=n_iter).detach().numpy().flatten()
    assert multi_mod_params == pytest.approx(data['params'], rel=0.1, abs=0.1)

    # Check that the actual predictions (fitted values) are nearly equal
    assert phenotype_mod_preds == pytest.approx(multi_mod_preds, rel=0.25, abs=0.25)
