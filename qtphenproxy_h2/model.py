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
        self.feature_names = feature_names if feature_names is not None else list(range(X.shape[1]))
        self.parameters_df = (
            pd.DataFrame(columns=['heritability_weight', 'seed', 'learning_rate', 'n_iter', *self.feature_names,
                                  'intercept'])
            .set_index(['heritability_weight', 'seed', 'learning_rate', 'n_iter'])
        )

    @classmethod
    def from_tables(cls, phenotype_code, genetic_covariance_matrix, phenotypic_covariance_matrix, phenotypes_df):
        """
        Load a model from a covariance matrices and phenotype data

        Parameters
        ----------
        phenotype_code : str
            Target phenotype of interest.
        genetic_covariance_matrix : pandas.DataFrame
            Square pandas.DataFrame whose index and columns contain `phenotype_code` and entries are equal to genetic
            covariances
        phenotypic_covariance_matrix : pandas.DataFrame
            Square pandas.DataFrame whose index and columns contain `phenotype_code` and entries are equal to phenotypic
            covariances
        phenotypes_df : pandas.DataFrame
            People x phenotypes table. Column names are the same as those for genetic and phenotypic covariance matrices
        """
        target_heritability = genetic_covariance_matrix.loc[phenotype_code, phenotype_code]
        feature_cols = phenotypes_df.columns.drop(phenotype_code)
        X = phenotypes_df.loc[:, feature_cols]
        y = phenotypes_df.loc[:, [phenotype_code]]

        feature_genetic_covariance = genetic_covariance_matrix.loc[feature_cols, feature_cols]
        feature_phenotypic_covariance = phenotypic_covariance_matrix.loc[feature_cols, feature_cols]

        target_genetic_covariance = genetic_covariance_matrix.loc[feature_cols, [phenotype_code]]
        target_phenotypic_covariance = phenotypic_covariance_matrix.loc[feature_cols, [phenotype_code]]

        # Convert all to torch tensors
        X = torch.from_numpy(X.values).float()
        y = torch.from_numpy(y.values).float()
        feature_genetic_covariance = torch.from_numpy(feature_genetic_covariance.values).float()
        feature_phenotypic_covariance = torch.from_numpy(feature_phenotypic_covariance.values).float()
        target_genetic_covariance = torch.from_numpy(target_genetic_covariance.values).float()
        target_phenotypic_covariance = torch.from_numpy(target_phenotypic_covariance.values).float()

        return cls(X=X, y=y, h2_target=target_heritability, feature_genetic_covariance=feature_genetic_covariance,
                   feature_phenotypic_covariance=feature_phenotypic_covariance,
                   target_genetic_covariance=target_genetic_covariance,
                   target_phenotypic_covariance=target_phenotypic_covariance, feature_names=feature_cols.tolist())

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
        """
        Fit a model for a single heritability weight/seed/learning rate/iteration combo

        Parameters
        ----------
        heritability_weight : float, optional
            Weight applied to the heritability term in the model loss function, by default 0
        seed : int, optional
            Random seed used for training, by default 0
        learning_rate : float, optional
            Learning rate used for training, by default 0.001
        n_iter : int, optional
            Number of training iterations, by default 5000
        verbose : bool, optional
            Whether to print training loss progress, by default False
        log_freq : int, optional
            Number of iterations between logging during training, by default 100
        """
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
        best_setting = (
            self.hyperparameter_log_df
            .reset_index()
            .loc[lambda df: df['qt_metric'] == df['qt_metric'].min()]
            .to_dict('records')[0]
        )
        return (best_setting['heritability_weight'], best_setting['seed'], best_setting['learning_rate'],
                best_setting['n_iter'])

    def get_predictions(self, heritability_weight, seed, learning_rate, n_iter):
        """Generate predicted values for all samples for a given train setting"""
        parameters = self.parameters_df.loc[(heritability_weight, seed, learning_rate, n_iter)].values
        weights = torch.from_numpy(parameters[:-1]).float()
        intercept = torch.tensor((parameters[-1])).float()
        model = torch.nn.Linear(in_features=self.X.shape[1], out_features=1, bias=True)
        model.weights, model.intercept = (weights, intercept)
        return model(self.X)

    def save_fit(self, path, person_ids=None, save_raw_data=True, overwrite=False):
        """Save a model into a new directory"""
        path = pathlib.Path(path)
        path.mkdir(exist_ok=overwrite)

        # Save the data used for hyperparameter optimization
        (
            self.hyperparameter_log_df
            .reset_index()
            .to_csv(path.joinpath('hyperparameter_log.tsv'), sep='\t', index=False)
        )

        # Save the actual trained parameter values
        (
            self.parameters_df
            .reset_index()
            .to_csv(path.joinpath('parameter_values.tsv.gz'), sep='\t', index=False, compression='gzip')
        )

        # Save the train log data
        self.train_log_df.to_csv(path.joinpath('training_log.tsv.gz'), sep='\t', index=False, compression='gzip')

        # Save feature names
        with open(path.joinpath('feature_names.tsv'), 'w+') as f:
            names_lines = [str(name) + '\n' for name in self.feature_names]
            f.writelines(names_lines)

        # Save the best model to a plink file
        if self.hyperparameter_log_df.shape[0] > 0:
            if person_ids is None:
                raise ValueError("Person IDs must be given to save predictions to a Plink format file")
            best_settings = self.get_best_setting()
            target = self.y.detach().numpy().flatten()
            best_wo_h2 = (
                self.hyperparameter_log_df
                .reset_index()
                .query('heritability_weight == 0')
                .loc[lambda df: df['qt_metric'] == df['qt_metric'].min()]
                .to_dict('records')[0]
            )
            best_wo_h2 = (best_wo_h2['heritability_weight'], best_wo_h2['seed'], best_wo_h2['learning_rate'],
                          best_wo_h2['n_iter'])
            no_h2_predictions = self.get_predictions(*best_wo_h2).detach().numpy().flatten()
            predictions = self.get_predictions(*best_settings).detach().numpy().flatten()
            plink_df = pd.DataFrame({'FID': person_ids, 'IID': person_ids, 'target': target,
                                     'qtphenproxy': no_h2_predictions, 'qtphenproxy_h2': predictions})
            plink_df.to_csv(path.joinpath('predictions.pheno'), sep='\t', index=False, header=True)

        if save_raw_data:
            raw_folder = path.joinpath('raw/')
            raw_folder.mkdir(exist_ok=True)
            torch.save(self.X, raw_folder.joinpath('X.pt'))
            torch.save(self.y, raw_folder.joinpath('y.pt'))

            torch.save(self.feature_g_cov, raw_folder.joinpath('feature_genetic_covariance.pt'))
            torch.save(self.feature_p_cov, raw_folder.joinpath('feature_phenotypic_covariance.pt'))
            torch.save(self.target_g_cov, raw_folder.joinpath('target_genetic_covariance.pt'))
            torch.save(self.target_p_cov, raw_folder.joinpath('target_phenotypic_covariance.pt'))

            with open(raw_folder.joinpath('h2.txt'), 'w+') as f:
                f.write(str(self.h2_target))
                f.write('\n')

    @classmethod
    def load_fit(cls, path):
        """Load an already fit model from the specified path"""
        path = pathlib.Path(path)
        X = torch.load(path.joinpath('raw/X.pt'))
        y = torch.load(path.joinpath('raw/y.pt'))
        feature_g_cov = torch.load(path.joinpath('raw/feature_genetic_covariance.pt'))
        feature_p_cov = torch.load(path.joinpath('raw/feature_phenotypic_covariance.pt'))
        target_g_cov = torch.load(path.joinpath('raw/target_genetic_covariance.pt'))
        target_p_cov = torch.load(path.joinpath('raw/target_phenotypic_covariance.pt'))
        with open(path.joinpath('raw/h2.txt'), 'r') as f:
            h2_target = float(f.readline().strip())
        cl = cls(X=X, y=y, h2_target=h2_target,  feature_genetic_covariance=feature_g_cov,
                 feature_phenotypic_covariance=feature_p_cov, target_genetic_covariance=target_g_cov,
                 target_phenotypic_covariance=target_p_cov)

        cl.hyperparameter_log_df = (
            pd.read_csv(path.joinpath('hyperparameter_log.tsv'), sep='\t')
            .set_index(['heritability_weight', 'seed', 'learning_rate', 'n_iter'])
        )
        cl.parameters_df = (
            pd.read_csv(path.joinpath('parameter_values.tsv.gz'), sep='\t')
            .set_index(['heritability_weight', 'seed', 'learning_rate', 'n_iter'])
        )
        cl.train_log_df = pd.read_csv(path.joinpath('training_log.tsv.gz'), sep='\t')
        cl.feature_names = pd.read_csv(path.joinpath('feature_names.tsv'), header=None).values.flatten().tolist()
        return cl


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


class CombinationFitter(MultiFitter):
    def fit_single_multiplier(self, multiplier=0.5, seed=0, learning_rate=0.001, n_iter=5000, verbose=False, log_freq=100):
        """Fit a model using a heritability weight, chosen as the relative size of the heritability loss compared to the
        overall training loss. Weight = multiplier * overall loss / proxy trait heritability"""
        self.fit_single(heritability_weight=0, seed=seed, learning_rate=learning_rate, n_iter=n_iter,
                        verbose=verbose, log_freq=log_freq)
        qt_metric = self.hyperparameter_log_df.loc[(0, seed, learning_rate, n_iter), 'qt_metric'].item()
        loss = (
            self.train_log_df
            .set_index(['heritability_weight', 'seed', 'learning_rate', 'n_iter', 'step'])
            .loc[(0, seed, learning_rate, n_iter, (n_iter - 1) - (n_iter - 1) % log_freq)]
            ['loss']
            .item()
        )
        multiplier_term = max(1 - multiplier, 1e-6)
        heritability_weight = multiplier * loss / (qt_metric * multiplier_term)
        assert isinstance(heritability_weight, float)
        self.fit_single(heritability_weight=heritability_weight, seed=seed, learning_rate=learning_rate, n_iter=n_iter,
                        verbose=verbose, log_freq=log_freq)

    def fit_binary_search(self, min_weight=0, max_weight=5, search_depth=10, seed=0, learning_rate=0.001, n_iter=5000,
                          verbose=False, log_freq=100):
        """Use binary search across heritability weights to find an approximate best solution"""
        for _ in tqdm.auto.trange(search_depth):
            x1 = min_weight + (max_weight - min_weight) / 3
            x2 = min_weight + 2 * (max_weight - min_weight) / 3
            self.fit_single(heritability_weight=x1, seed=seed, learning_rate=learning_rate, n_iter=n_iter,
                            verbose=verbose, log_freq=log_freq)
            self.fit_single(heritability_weight=x2, seed=seed, learning_rate=learning_rate, n_iter=n_iter,
                            verbose=verbose, log_freq=log_freq)
            y1 = self.hyperparameter_log_df.loc[(x1, seed, learning_rate, n_iter), 'qt_metric'].item()
            y2 = self.hyperparameter_log_df.loc[(x2, seed, learning_rate, n_iter), 'qt_metric'].item()
            if y1 < y2:
                max_weight = min_weight + (max_weight - min_weight) / 2
            else:
                min_weight = min_weight + (max_weight - min_weight) / 2

    def fit(self, n_orders_of_magnitude=6, binary_search_depth=10, seed=0, learning_rate=0.001, n_iter=5000, verbose=False, log_freq=100):
        """
        Fit the heritability-weighted QTPhenProxy model's hyperparameter using a two-stage method. First, fit models
        using weights of varying orders-of-magnitude to determine the best order of magnitude for the weight relative
        to the loss. Second, use binary search to explore that order of magnitude to find an approximate best
        hyperparameter.

        Parameters
        ----------
        binary_search_depth : int, optional
            Number of binary search iterations for exploring the appropriate order of magnitude, by default 10
        seed : int, optional
            Random seed for QTPhenProxy model fitting, by default 0
        learning_rate : float, optional
            Learning rate for QTPhenProxy model fitting, by default 0.001
        n_iter : int, optional
            Number of training iterations for the QTPhenProxy model, by default 5000
        verbose : bool, optional
            Whether to QTPhenProxy model print training information, by default False
        log_freq : int, optional
            Number of iterations at which to log QTPhenProxy model training statistics, by default 100
        """
        orders_of_magnitude = 10.0 ** - np.arange(n_orders_of_magnitude)
        for multiplier in orders_of_magnitude:
            self.fit_single_multiplier(multiplier=multiplier, seed=seed, learning_rate=learning_rate, n_iter=n_iter,
                                       verbose=verbose, log_freq=log_freq)

        best_magnitude = (
            self.hyperparameter_log_df
            .reset_index()
            .query('heritability_weight != 0')
            .loc[lambda df: df['qt_metric'] == df['qt_metric'].min(), 'heritability_weight']
            .item()
        )
        self.fit_binary_search(min_weight=best_magnitude / 10, max_weight=best_magnitude * 10,
                               search_depth=binary_search_depth, seed=seed, learning_rate=learning_rate, n_iter=n_iter,
                               verbose=verbose, log_freq=log_freq)
