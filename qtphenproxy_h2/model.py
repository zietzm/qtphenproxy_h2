import pandas as pd
import torch.nn


class PhenotypeFit



class PhenotypeFit(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heritability_weight=0, mse_weight=1,
                 genetic_covariance=None, phenotypic_covariance=None,
                 genetic_covariance_vector=None, phenotypic_covariance_vector=None):
        super(PhenotypeFit, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

        # Information stored for use in the loss function
        self.h2_weight = heritability_weight
        self.mse_weight = mse_weight
        self.g_cov = genetic_covariance
        self.p_cov = phenotypic_covariance
        self.g_cov_vec = genetic_covariance_vector
        self.p_cov_vec = phenotypic_covariance_vector

    def forward(self, x):
        return self.linear(x)

    def heritability(self):
        return heritability_fn(self.linear.weight, self.g_cov, self.p_cov)

    def coheritability(self):
        return coheritability_fn(self.linear.weight, self.g_cov_vec, self.p_cov_vec)

    def loss_fn(self, output, target, weights):
        return self.mse_weight * mse_loss(output, target) - self.h2_weight * self.heritability()

    def fit(self, X, y, n_iter, learning_rate, seed, verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for step in range(n_iter):
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = self.loss_fn(outputs, y, self.linear.weight)
            loss.backward()
            optimizer.step()

            if verbose and step % 1000 == 0:
                print(f'Training loss: {loss.item():.3f}')


class QTPhenProxyElbowFitter:
    def __init__(self, h2_target, X, y, feature_genetic_covariance, feature_phenotypic_covariance,
                 target_genetic_covariance, target_phenotypic_covariance):
        self.h2_target = h2_target
        self.X = X
        self.y = y
        self.feature_g_cov = feature_genetic_covariance
        self.feature_p_cov = feature_phenotypic_covariance
        self.target_g_cov = target_genetic_covariance
        self.target_p_cov = target_phenotypic_covariance

    def fit_individual(self, h2_weight, n_iter, learning_rate, seed, verbose=False):
        fitter = PhenotypeFit(input_dim=self.X.shape[1], output_dim=self.y.shape[1],
                              heritability_weight=h2_weight, mse_weight=1,
                              genetic_covariance=self.feature_g_cov,
                              phenotypic_covariance=self.feature_p_cov,
                              genetic_covariance_vector=self.target_g_cov,
                              phenotypic_covariance_vector=self.target_p_cov)
        fitter.fit(X=self.X, y=self.y, n_iter=n_iter, learning_rate=learning_rate,
                   seed=seed, verbose=verbose)
        metric = qt_metric(target_heritability=self.h2_target, weights=fitter.linear.weight,
                           feature_genetic_covariance=self.feature_g_cov,
                           feature_phenotypic_covariance=self.feature_p_cov,
                           target_genetic_covariance=self.target_g_cov,
                           target_phenotypic_covariance=self.target_p_cov)
        return metric, fitter.linear.weight.detach()

    def fit_all(self, h2_weights, n_iter=5000, learning_rate=0.0008, seed=0, verbose=False):
        log = {'h2_weight': list(), 'qt_metric': list(), 'feature_weights': list()}
        for h2_weight in h2_weights:
            metric, feature_weights = self.fit_individual(h2_weight=h2_weight, n_iter=n_iter,
                                                          learning_rate=learning_rate, seed=seed,
                                                          verbose=verbose)
            log['h2_weight'].append(h2_weight)
            log['qt_metric'].append(metric)
            log['feature_weights'].append(feature_weights)
        return pd.DataFrame(log)

    def fit(self, h2_weights, n_iter=5000, learning_rate=0.0008, seed=0, verbose=False):
        log_df = fit_all(h2_weights=h2_weights, n_iter=n_iter, learning_rate=learning_rate,
                         seed=seed, verbose=verbose)
        best_setting = log_df[log_df['qt_metric'] == log_df['qt_metric'].min()]

        return