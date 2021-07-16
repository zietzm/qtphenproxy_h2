import pathlib

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
    def __init__(self, input_dim, output_dim=1, mse_weight=1, heritability_weight=0,
                 feature_genetic_covariance=None, feature_phenotypic_covariance=None,
                 target_genetic_covariance=None, target_phenotypic_covariance=None):
        """
        Heritability-weighted QTPhenProxy model class. Used to fit a model with
        a single hyperparameter setting. The heritability weight refers to the
        loss function weight applied to the heritability of the predictions.

        Parameters
        ----------
        input_dim : int
            Number of feature traits
        output_dim : int
            Number of target traits, by default 1
        mse_weight : float, optional
            Weight applied to the MSE term of the loss function, by default 1
        heritability_weight : float, optional
            Weight applied to the heritability term of the loss function, by default 0
        feature_genetic_covariance : torch.tensor, optional
            Matrix of genetic covariances for the feature traits, (n_features x n_features), by default None
        feature_phenotypic_covariance : torch.tensor, optional
            Matrix of phenotypic covariances for the feature traits, (n_features x n_features), by default None
        target_genetic_covariance : torch.tensor, optional
            Vector of genetic covariances between the feature and target traits, (n_features x 1), by default None
        target_phenotypic_covariance : torch.tensor, optional
            Vector of phenotypic covariances between the feature and target traits, (n_features x 1), by default None
        """
        super(PhenotypeFit, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)

        # Information stored for use in the loss function
        self.h2_weight = heritability_weight
        self.mse_weight = mse_weight
        self.feature_g_cov = feature_genetic_covariance
        self.feature_p_cov = feature_phenotypic_covariance
        self.target_g_cov = target_genetic_covariance
        self.target_p_cov = target_phenotypic_covariance

    def forward(self, x):
        return self.linear(x)

    def heritability(self):
        """Compute the heritability of the fitted trait"""
        return heritability_fn(self.linear.weight, self.feature_g_cov, self.feature_p_cov)

    def coheritability(self):
        """Compute the coheritability between the fitted and target traits"""
        return coheritability_fn(self.linear.weight, self.target_g_cov, self.target_p_cov)

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
        """
        Heritability weighted QTPhenProxy model fitter. This class fits multiple
        models to find a good value for the heritability weight hyperparameter.

        Parameters
        ----------
        X : torch.tensor
            Feature phenotype values (n_samples x n_features)
        y : torch.tensor
            Target phenotype values (n_samples x 1)
        h2_target : float
            Heritability of the target phenotype, for hyperparameter optimization.
        feature_genetic_covariance : torch.tensor
            Matrix of genetic covariances for the feature traits. (n_features x n_features)
        feature_phenotypic_covariance : torch.tensor
            Matrix of phenotypic covariances for the feature traits. (n_features x n_features)
        target_genetic_covariance : torch.tensor
            Vector of genetic covariances between the feature and target traits. (n_features x 1)
        target_phenotypic_covariance : torch.tensor
            Vector of phenotypic covariances between the feature and target traits. (n_features x 1)
        """
        self.X = X
        self.y = y
        self.h2_target = h2_target
        self.feature_g_cov = feature_genetic_covariance
        self.feature_p_cov = feature_phenotypic_covariance
        self.target_g_cov = target_genetic_covariance
        self.target_p_cov = target_phenotypic_covariance
        self.is_trained = False
        self.log_df = None
        self.setting_to_params = None

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

        self.setting_to_params = dict()
        for setting in iterator:
            model = PhenotypeFit(
                input_dim=self.X.shape[1], output_dim=1, mse_weight=1, heritability_weight=setting['heritability_weight'],
                feature_genetic_covariance=self.feature_g_cov, feature_phenotypic_covariance=self.feature_p_cov,
                target_genetic_covariance=self.target_g_cov, target_phenotypic_covariance=self.target_p_cov
            )
            # Train the model
            model.fit(X=self.X, y=self.y, n_iter=n_iter, learning_rate=setting['learning_rate'], seed=setting['seed'],
                      verbose=verbose)

            # Gather model weights
            self.setting_to_params[(setting['heritability_weight'], setting['seed'])] = tuple(model.linear.parameters())

            # Compute the metric for hyperparameter optimization
            setting['qt_metric'] = self.qt_metric(model.linear.weight)

        self.is_trained = True
        self.log_df = pd.DataFrame(train_settings)

    def get_best_setting(self):
        """Return the hyperparameter setting that optimized the QT metric"""
        if not self.is_trained:
            raise ValueError("The models must be trained before predictions can be generated.")
        best_setting = (
            self.log_df
            .loc[self.log_df['qt_metric'] == self.log_df['qt_metric'].min()]
            .to_dict('records')[0]
        )
        return (best_setting['heritability_weight'], best_setting['seed'])

    def get_predictions(self, heritability_weight, seed):
        """Generate predicted values for all samples for a given train setting"""
        if not self.is_trained:
            raise ValueError("The models must be trained before predictions can be generated.")
        parameters = self.setting_to_params[(heritability_weight, seed)]
        model = torch.nn.Linear(in_features=self.X.shape[1], out_features=1, bias=True)
        model.weight, model.bias = parameters
        return model(self.X)

    def save_fit(self, path, feature_names=None):
        """Save a model into a new directory"""
        path = pathlib.Path(path)
        path.mkdir(exist_ok=False)
        self.log_df.to_csv(path.joinpath('train_log.tsv'), sep='\t', index=False)

        # Flatten {(int, int): (torch.tensor, torch.tensor)} to a [float, int, float, ..., float]
        settings_parameters = [(*key, *wt.detach().flatten().tolist(), intercept.item()) for key, (wt, intercept) in self.setting_to_params.items()]

        if feature_names is None:
            feature_names = (f'feature_{i}' for i in range(self.X.shape[1]))
        colnames = ['heritability_weight', 'seed', *feature_names, 'intercept']
        settings_parameters_df = pd.DataFrame(settings_parameters, columns=colnames)
        settings_parameters_df.to_csv(path.joinpath('parameter_values.tsv.gz'), sep='\t', index=False, compression='gzip')
