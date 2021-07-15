import numpy as np
import pandas as pd
import torch
import tqdm.auto


def heritability_fn(weights, feature_genetic_covariance, feature_phenotypic_covariance):
    """Computes the heritability of a derived trait which is a linear combination of traits
    with known heritabilities and co-heritabilities."""
    return (weights @ feature_genetic_covariance @ weights.T) / \
        (weights @ feature_phenotypic_covariance @ weights.T)

def coheritability_fn(weights, target_genetic_covariance, target_phenotypic_covariance):
    """Computes the coheritability of a trait with a linear combination of other traits with
    known co-heritabilities with the target."""
    return (weights @ target_genetic_covariance) / (weights @ target_phenotypic_covariance)

def mse_loss(output, target):
    """Mean squared error"""
    return torch.mean((output - target)**2)


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

    def loss_fn(self, output, target):
        return self.mse_weight * mse_loss(output, target) - self.h2_weight * self.heritability()

    def fit(self, X, y, n_iter, learning_rate, seed, verbose=False):
        """
        Fit the specified QTPhenProxy model

        Parameters
        ----------
        X : torch.tensor
            Individual-level data representing the feature traits (samples x features).
        y : torch.tensor
            Individual-level data for the target phenotype (samples x 1).
        n_iter : int
            Number of training iterations
        learning_rate : float
            Learning rate for the Adam optimizer
        seed : int
            Random seed for training
        verbose : bool, optional
            Whether to print loss every 1000 training steps, by default False
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        torch.manual_seed(seed)
        for step in range(n_iter):
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = self.loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            if verbose and step % 1000 == 0:
                print(f'Training loss: {loss.item():.3f}')


class MultiHeritabilityQTPhenProxy:
    def __init__(self, X, y, h2_target,
                 feature_genetic_covariance, feature_phenotypic_covariance,
                 target_genetic_covariance, target_phenotypic_covariance):
        self.X = X
        self.y = y
        self.h2_target = h2_target
        self.feature_g_cov = feature_genetic_covariance
        self.feature_p_cov = feature_phenotypic_covariance
        self.target_g_cov = target_genetic_covariance
        self.target_p_cov = target_phenotypic_covariance
        self.log_df = None
        self.setting_to_prediction = None

    def qt_metric(self, weights):
        """
        Computes the QTPhenProxy evaluation metric ("QT metric"), which is used for hyperparameter
        optimization in QTPhenProxy models.

        The metric is minimized when a derived trait is exactly equal to the genetic component of
        the target trait.

        Let y be the target trait (data). Let ŷ be the derived trait (some function of feature traits).
        h2(x) is the heritability of trait x, and h2(x, z) is the coheritability of traits x and z.

        |h2(y) — h2(y, ŷ)| + |1 — h2(ŷ)|
        """
        ch2 = coheritability_fn(weights, self.target_g_cov, self.target_p_cov).item()
        term_1 = np.abs(self.h2_target - ch2)

        h2 = heritability_fn(weights, self.feature_g_cov, self.feature_p_cov).item()
        term_2 = np.abs(1 - h2)
        return term_1 + term_2

    def fit(self, heritability_weights, n_seeds, n_iter, learning_rate, verbose=False):
        """
        Fit a heritability-weighted QTPhenProxy model of a trait.

        Parameters
        ----------
        heritability_weights : iterable of float
            Heritability weights to try
        n_seeds : int
            Number of random seeds to try for each heritability weight
        n_iter : int
            Number of training iterations to use
        learning_rate : float
        verbose : bool, optional
            Whether to print a progress bar and track training performance, by default False

        Returns
        -------
        torch.tensor
            Individual QTPhenProxy predictions from the best model fit
        """
        train_settings = [
            {
                'heritability_weight': wt,
                'seed': seed,
                'n_iter': n_iter,
                'learning_rate': learning_rate,
                'verbose': verbose
            }
            for wt in heritability_weights for seed in range(n_seeds)
        ]

        if verbose:
            iterator = tqdm.auto.tqdm(train_settings)
        else:
            iterator = train_settings

        self.setting_to_prediction = dict()
        for setting in iterator:
            model = PhenotypeFit(
                input_dim=self.X.shape[1], output_dim=1, heritability_weight=setting['heritability_weight'],
                mse_weight=1, genetic_covariance=self.feature_g_cov, phenotypic_covariance=self.feature_p_cov,
                genetic_covariance_vector=self.target_g_cov, phenotypic_covariance_vector=self.target_p_cov
            )
            # Train the model
            model.fit(X=self.X, y=self.y, n_iter=n_iter, learning_rate=setting['learning_rate'], seed=setting['seed'],
                      verbose=verbose)

            # Gather predictions
            output = model(self.X)
            self.setting_to_prediction[(setting['heritability_weight'], setting['seed'])] = output

            # Compute the metric for hyperparameter optimization
            setting['qt_metric'] = self.qt_metric(model.linear.weight)

        self.log_df = pd.DataFrame(train_settings)

        best_setting = (
            self.log_df
            .loc[self.log_df['qt_metric'] == self.log_df['qt_metric'].min()]
            .to_dict('records')[0]
        )
        best_predictions = self.setting_to_prediction[(best_setting['heritability_weight'], best_setting['seed'])]
        return best_predictions
