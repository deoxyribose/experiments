from models_and_guides import *

class DAGModel(Model):
    """
    Template class for code generated from a DAG
    """
    def __init__(self, X, K, C, batch_size, _id):
        self.K = K
        self.C = C
        super(DAGModel, self).__init__(X, batch_size, _id)


    def get_param_shapes_and_support(self, _id=None):
        if _id == None:
            _id = self._id
        return {f'mixing_proportions_concentration_init_{_id}': ((self.C,),
            constraints.positive),
            f'mixing_proportions_prior_concentration_init_{_id}': ((self.C,),
            constraints.positive), f'cov_factor_scale_init_{_id}': ((self.K,
            self.D), constraints.positive),
            f'cov_factor_prior_scale_init_{_id}': ((self.K, self.D),
            constraints.positive), f'cov_factor_loc_init_{_id}': ((self.C,self.K, self
            .D), constraints.real), f'cov_factor_prior_loc_init_{_id}': ((self.C,self.
            K, self.D), constraints.real), f'cov_diag_scale_init_{_id}': ((self
            .D,), constraints.positive), f'cov_diag_prior_scale_init_{_id}': ((
            self.D,), constraints.positive), f'cov_diag_loc_init_{_id}': ((self
            .D,), constraints.real), f'cov_diag_prior_loc_init_{_id}': ((self.D
            ,), constraints.real)}


    def model(self, X):
        K = self.K
        C = self.C
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
        mixing_proportions_prior_concentration_init = self.param_init[
            f'mixing_proportions_prior_concentration_init_{_id}']
        mixing_proportions_prior_concentration = pyro.param(
            f'mixing_proportions_prior_concentration_{_id}',
            mixing_proportions_prior_concentration_init, constraint=constraints
            .positive)
        mixing_proportions = pyro.sample(f'mixing_proportions_{_id}', dist.
            Dirichlet(mixing_proportions_prior_concentration))
        with pyro.plate(f'C_{_id}', C):
            with pyro.plate(f'CD_{_id}', D):
                with pyro.plate(f'CDK_{_id}', K):
                    cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(
                        cov_factor_prior_loc, cov_factor_prior_scale))
                cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(
                    cov_diag_prior_loc, cov_diag_prior_scale))
            cov_factor_T = torch.transpose(cov_factor, 0, 1)
        with pyro.plate(f'N_{_id}', N, subsample_size=self.batch_size) as ind:
            assignment = pyro.sample(f'assignment_{_id}', dist.Categorical(
                mixing_proportions))
            loc_idx = torch.index_select(prior_loc, assignment, 0)
            cov_factor_T_idx = torch.index_select(cov_factor_T, assignment, 0)
            cov_diag_idx = torch.index_select(cov_diag, assignment, 0)
            X = pyro.sample('obs', dist.LowRankMultivariateNormal(loc_idx,
                cov_factor_T_idx, cov_diag_idx), obs=X.index_select(0, ind))
        return X


    def guide(self, X):
        K = self.K
        C = self.C
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
        loc = torch.zeros(D)
        mixing_proportions_concentration_init = self.param_init[
            f'mixing_proportions_concentration_init_{_id}']
        mixing_proportions_concentration = pyro.param(
            f'mixing_proportions_concentration_{_id}',
            mixing_proportions_concentration_init, constraint=constraints.positive)
        mixing_proportions = pyro.sample(f'mixing_proportions_{_id}', dist.
            Dirichlet(mixing_proportions_concentration))
        with pyro.plate(f'N_{_id}', N, subsample_size=self.batch_size) as ind:
            assignment = pyro.sample(f'assignment_{_id}', dist.Categorical(
                mixing_proportions))
        with pyro.plate(f'C_{_id}', C):
            with pyro.plate(f'CD_{_id}', D):
                with pyro.plate(f'CDK_{_id}', K):
                    cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(
                        cov_factor_loc, cov_factor_scale))
                cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(
                    cov_diag_loc, cov_diag_scale))
            cov_factor_T = torch.transpose(cov_factor, 0, 1)
        return cov_factor, cov_diag, mixing_proportions, assignment
