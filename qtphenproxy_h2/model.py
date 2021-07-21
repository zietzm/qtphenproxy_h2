from os import nice
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

        self.train_log_df = None

    def forward(self, x):
        return self.linear(x)

    def heritability(self, weights):
        """Compute the heritability of the fitted trait"""
        return heritability_fn(weights, self.feature_g_cov, self.feature_p_cov)

    def coheritability(self, weights):
        """Compute the coheritability between the fitted and target traits"""
        return coheritability_fn(weights, self.target_g_cov, self.target_p_cov)

    def loss_fn(self, output, target, weights):
        return self.mse_weight * mse_loss(output, target) - self.h2_weight * self.heritability(weights)

    def fit(self, X, y, n_iter, learning_rate, seed, verbose=False, log_freq=100):
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
        log_freq : int, optional
            How many steps should pass before the training loss is logged
        """
        train_log = list()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        torch.manual_seed(seed)
        for step in range(n_iter):
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = self.loss_fn(outputs, y, self.linear.weight)
            loss.backward()
            optimizer.step()

            if step % log_freq == 0:
                train_log.append([step, loss.item()])

            if verbose and step % 1000 == 0:
                print(f'Training loss: {loss.item():.3f}')
        self.train_log_df = pd.DataFrame(train_log, columns=['step', 'loss'])


class MultiFitter:
    def __init__(self, X, y, h2_target, feature_genetic_covariance, feature_phenotypic_covariance,
                 target_genetic_covariance, target_phenotypic_covariance, feature_names=None):
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
        feature_names : List[string]
            List of names for each feature, in the same order as the features appear, optional
        """
        self.X = X
        self.y = y
        self.h2_target = h2_target
        self.feature_g_cov = feature_genetic_covariance
        self.feature_p_cov = feature_phenotypic_covariance
        self.target_g_cov = target_genetic_covariance
        self.target_p_cov = target_phenotypic_covariance
        self.hyperparameter_log_df = (
            pd.DataFrame(columns=['heritability_weight', 'seed', 'learning_rate', 'n_iter', 'qt_metric'])
            .set_index(['heritability_weight', 'seed', 'learning_rate', 'n_iter'])
        )
        self.train_log_df = pd.DataFrame(columns=['heritability_weight', 'seed', 'learning_rate', 'n_iter', 'step',
                                                  'loss'])
        self.parameters_df = pd.DataFrame()  #  heritability_weight	seed	learning_rate   n_iter
        self.feature_names = feature_names if feature_names is not None else list(range(X.shape[1]))

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

    def fit_single(self, heritability_weight=0, seed=0, learning_rate=0.001, n_iter=5000, verbose=False, log_freq=100):
        if (heritability_weight, seed, learning_rate, n_iter) in self.hyperparameter_log_df.index:
            return
        # Instantiate the model class
        model = PhenotypeFit(
            input_dim=self.X.shape[1], output_dim=1, mse_weight=1,
            heritability_weight=heritability_weight, feature_genetic_covariance=self.feature_g_cov,
            feature_phenotypic_covariance=self.feature_p_cov, target_genetic_covariance=self.target_g_cov,
            target_phenotypic_covariance=self.target_p_cov
        )

        # Train the model
        model.fit(X=self.X, y=self.y, n_iter=n_iter, learning_rate=learning_rate, seed=seed, verbose=verbose,
                  log_freq=log_freq)

        # Gather the model's training log
        model_train_df = (
            model.train_log_df
            .assign(heritability_weight=heritability_weight, seed=seed, learning_rate=learning_rate, n_iter=n_iter)
            .loc[:, ['heritability_weight', 'seed', 'learning_rate', 'n_iter', 'step', 'loss']]
        )
        self.train_log_df = pd.concat([self.train_log_df, model_train_df], ignore_index=True)

        # Compute the model's QT metric
        model_qt_metric = self.qt_metric(model.linear.weight)
        model_qt_metric_row = (
            pd.DataFrame({
                'heritability_weight': [heritability_weight], 'seed': [seed], 'learning_rate': [learning_rate],
                'n_iter': [n_iter], 'qt_metric': [model_qt_metric]
            })
            .set_index(['heritability_weight', 'seed', 'learning_rate', 'n_iter'])
        )
        self.hyperparameter_log_df = pd.concat([self.hyperparameter_log_df, model_qt_metric_row], ignore_index=False)

        # Gather the model's weights
        weights, intercept = tuple(model.linear.parameters())
        weights = weights.detach().flatten().tolist()
        intercept = intercept.item()
        row = [heritability_weight, seed, learning_rate, n_iter, *weights, intercept]
        colnames = ['heritability_weight', 'seed', 'learning_rate', 'n_iter', *self.feature_names, 'intercept']
        model_parameters_df = (
            pd.DataFrame([row], columns=colnames)
            .set_index(['heritability_weight', 'seed', 'learning_rate', 'n_iter'])
        )
        self.parameters_df = pd.concat([self.parameters_df, model_parameters_df], ignore_index=False)

    def get_best_setting(self):
        """Return the hyperparameter setting that optimized the QT metric"""
        if not self.is_trained:
            raise ValueError("The models must be trained before predictions can be generated.")
        best_setting = (
            self.log_df
            .loc[self.log_df['qt_metric'] == self.log_df['qt_metric'].min()]
            .to_dict('records')[0]
        )
        return (best_setting['heritability_weight'], best_setting['seed'],
                best_setting['n_iter'], best_setting['learning_rate'])

    def get_predictions(self, heritability_weight, seed, n_iter, learning_rate):
        """Generate predicted values for all samples for a given train setting"""
        parameters = self.parameters_df.loc[(heritability_weight, seed, learning_rate, n_iter)].values
        weights = torch.from_numpy(parameters[:-1]).float()
        intercept = torch.tensor((parameters[-1])).float()
        model = torch.nn.Linear(in_features=self.X.shape[1], out_features=1, bias=True)
        model.weights, model.intercept = (weights, intercept)
        return model(self.X)

    def save_fit(self, path, overwrite=False):
        """Save a model into a new directory"""
        path = pathlib.Path(path)
        path.mkdir(exist_ok=overwrite)

        # Save the data used for hyperparameter optimization
        self.hyperparameter_log_df.to_csv(path.joinpath('hyperparameter_log.tsv'), sep='\t', index=False)

        # Save the actual trained parameter values
        (
            self.parameters_df
            .reset_index()
            .to_csv(path.joinpath('parameter_values.tsv.gz'), sep='\t', index=False, compression='gzip')
        )

        # Save the train log data
        self.train_log_df.to_csv(path.joinpath('training_log.tsv.gz'), sep='\t', index=False, compression='gzip')


class GradientDescentFitter(MultiFitter):
    def __init__(self, X, y, h2_target, feature_genetic_covariance, feature_phenotypic_covariance,
                 target_genetic_covariance, target_phenotypic_covariance, feature_names=None):
        super().__init__(X, y, h2_target, feature_genetic_covariance, feature_phenotypic_covariance,
                         target_genetic_covariance, target_phenotypic_covariance, feature_names=feature_names)
        self.gradient_descent_log_df = pd.DataFrame()

    def fit_seed(self, gd_lr=0.1, gd_n_iter=100, seed=0, learning_rate=0.001, n_iter=5000, verbose=True, log_freq=100):
        """
        Fit a QTPhenProxy model using gradient descent to optimize the heritability weight hyperparameter

        Parameters
        ----------
        gd_lr : float, optional
            Learning rate for hyperparameter optimization gradient descent, by default 0.1
        gd_n_iter : int, optional
            Number of gradient descent iterations for hyperparameter optimization, by default 100
        seed : int, optional
            Random seed for gradient descent, by default 0
        learning_rate : float, optional
            Learning rate for the heritability-weighted QTPhenProxy model itself, by default 0.001
        n_iter : int, optional
            Number of training iterations for individual QTPhenProxy models, by default 100
        verbose : bool, optional
            Whether to print information about training, by default True
        log_freq : int, optional
            Number of iterations at which to log training information, by default 100
        """
        # Initialize the gradient using the starting weights 0.5 and 1
        self.fit_single(heritability_weight=0, seed=seed, learning_rate=learning_rate, n_iter=n_iter,
                        verbose=verbose, log_freq=log_freq)
        self.fit_single(heritability_weight=0.1, seed=seed, learning_rate=learning_rate, n_iter=n_iter,
                        verbose=verbose, log_freq=log_freq)
        old_y = self.hyperparameter_log_df.loc[(0, seed, learning_rate, n_iter), 'qt_metric'].item()
        y = self.hyperparameter_log_df.loc[(0.1, seed, learning_rate, n_iter), 'qt_metric'].item()
        grad = (old_y - y) / (0.1)
        x = 0.1

        log = list()
        for _ in tqdm.auto.trange(gd_n_iter):
            if verbose:
                print(f"x: {x:.3f}; y: {y:.3f}; grad: {grad:.3f}")
            log.append((x, y, grad))
            old_x = x
            old_y = y
            x = old_x - gd_lr * grad
            x = round(x, ndigits=5)
            if x == old_x:
                break
            self.fit_single(heritability_weight=x, seed=seed, learning_rate=learning_rate, n_iter=n_iter,
                            verbose=verbose, log_freq=log_freq)
            y = self.hyperparameter_log_df.loc[(x, seed, learning_rate, n_iter), 'qt_metric'].item()
            grad = (old_y - y) / (old_x - x)
        log_df = (
            pd.DataFrame(log, columns=['heritability_weight', 'qt_metric', 'gradient'])
            .assign(seed=seed)
        )
        self.gradient_descent_log_df = pd.concat([self.gradient_descent_log_df, log_df])

    def fit(self, gd_lr=0.1, gd_n_iter=100, n_seeds=1, learning_rate=0.001, n_iter=5000, verbose=True, log_freq=100):
        """
        Fit a QTPhenProxy model using gradient descent to optimize the heritability weight hyperparameter using
        multiple random seeds

        Parameters
        ----------
        gd_lr : float, optional
            Learning rate for hyperparameter optimization gradient descent, by default 0.1
        gd_n_iter : int, optional
            Number of gradient descent iterations for hyperparameter optimization, by default 100
        n_seeds : int, optional
            Number of random seed for training hyperparameters, by default 0
        learning_rate : float, optional
            Learning rate for the heritability-weighted QTPhenProxy model itself, by default 0.001
        n_iter : int, optional
            Number of training iterations for individual QTPhenProxy models, by default 100
        verbose : bool, optional
            Whether to print information about training, by default True
        log_freq : int, optional
            Number of iterations at which to log training information, by default 100
        """
        for seed in range(n_seeds):
            self.fit_seed(gd_lr=gd_lr, gd_n_iter=gd_n_iter, seed=seed, learning_rate=learning_rate, n_iter=n_iter,
                          verbose=verbose, log_freq=log_freq)
