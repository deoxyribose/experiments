from models_and_guides import *

class DAGModel(Model):
    """
    Template class for code generated from a DAG
    """
    def __init__(self, X, K, batch_size, _id):
        self.K = K
        super(DAGModel, self).__init__(X, batch_size, _id)


    def get_param_shapes_and_support(self, _id=None):
        if _id == None:
            _id = self._id
        return {f'cov_factor_scale_init_{_id}': ((self.K, self.D), constraints.
            positive), f'cov_factor_prior_scale_init_{_id}': ((self.K, self.D),
            constraints.positive), f'cov_factor_loc_init_{_id}': ((self.K, self
            .D), constraints.real), f'cov_factor_prior_loc_init_{_id}': ((self.
            K, self.D), constraints.real), f'cov_diag_scale_init_{_id}': ((self
            .D,), constraints.positive), f'cov_diag_prior_scale_init_{_id}': ((
            self.D,), constraints.positive), f'cov_diag_loc_init_{_id}': ((self
            .D,), constraints.real), f'cov_diag_prior_loc_init_{_id}': ((self.D
            ,), constraints.real)}


    def model(self, X):
        K = self.K
        N, D = X.shape
        _id = self._id
        cov_diag_prior_loc_init = self.param_init[f'cov_diag_prior_loc_init_{_id}']
        cov_diag_prior_loc = pyro.param(f'cov_diag_prior_loc_{_id}',
            cov_diag_prior_loc_init)
        cov_diag_prior_scale_init = self.param_init[
            f'cov_diag_prior_scale_init_{_id}']
        cov_diag_prior_scale = pyro.param(f'cov_diag_prior_scale_{_id}',
            cov_diag_prior_scale_init, constraint=constraints.positive)
        cov_factor_prior_loc_init = self.param_init[
            f'cov_factor_prior_loc_init_{_id}']
        cov_factor_prior_loc = pyro.param(f'cov_factor_prior_loc_{_id}',
            cov_factor_prior_loc_init)
        cov_factor_prior_scale_init = self.param_init[
            f'cov_factor_prior_scale_init_{_id}']
        cov_factor_prior_scale = pyro.param(f'cov_factor_prior_scale_{_id}',
            cov_factor_prior_scale_init, constraint=constraints.positive)
        prior_loc = torch.zeros(D)
        with pyro.plate(f'D_{_id}', D):
            with pyro.plate(f'DK_{_id}', K):
                cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(
                    cov_factor_prior_loc, cov_factor_prior_scale))
            cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(
                cov_diag_prior_loc, cov_diag_prior_scale))
        with pyro.plate(f'N_{_id}', N, subsample_size=self.batch_size) as ind:
            X = pyro.sample('obs', dist.LowRankMultivariateNormal(prior_loc,
                cov_factor, cov_diag), obs=X.index_select(0, ind))
        return X


    def guide(self, X):
        K = self.K
        N, D = X.shape
        _id = self._id
        cov_diag_loc_init = self.param_init[f'cov_diag_loc_init_{_id}']
        cov_diag_loc = pyro.param(f'cov_diag_loc_{_id}', cov_diag_loc_init)
        cov_diag_scale_init = self.param_init[f'cov_diag_scale_init_{_id}']
        cov_diag_scale = pyro.param(f'cov_diag_scale_{_id}',
            cov_diag_scale_init, constraint=constraints.positive)
        cov_factor_loc_init = self.param_init[f'cov_factor_loc_init_{_id}']
        cov_factor_loc = pyro.param(f'cov_factor_loc_{_id}', cov_factor_loc_init)
        cov_factor_scale_init = self.param_init[f'cov_factor_scale_init_{_id}']
        cov_factor_scale = pyro.param(f'cov_factor_scale_{_id}',
            cov_factor_scale_init, constraint=constraints.positive)
        with pyro.plate(f'D_{_id}', D):
            with pyro.plate(f'DK_{_id}', K):
                cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(
                    cov_factor_loc, cov_factor_scale))
            cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(
                cov_diag_loc, cov_diag_scale))
        return cov_factor, cov_diag
