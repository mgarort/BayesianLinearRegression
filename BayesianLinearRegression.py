import numpy as np
import torch
import torch.nn as nn


class BayesianLinearRegression(nn.Module):  
    
    def __init__(self, X_train, Y_train, jitter_factor=1e-12):      
        super(BayesianLinearRegression, self).__init__()
        self.X_train = X_train  # inputs must be tensors
        self.Y_train = Y_train
        self.jitter_factor = torch.Tensor([jitter_factor]).type(torch.double)
        self.n_samples = X_train.shape[0]
        self.n_features = X_train.shape[1]
        # initialize hyperparameters
        self.log_alpha = nn.Parameter(torch.Tensor([1]).type(torch.double))
        self.log_beta = nn.Parameter(torch.Tensor([1]).type(torch.double))


    def forward(self, X):
        A = self.alpha * torch.eye(self.n_features).type(torch.double) + self.beta * self.X_train.t() @ self.X_train
        L_A = torch.cholesky(self.jitter(A))
        H1T_star = torch.triangular_solve(X.t(), L_A, upper=False)[0] # Have to compute the transpose because of the way triangular_solve/trtrs is written
        H1T_train = torch.triangular_solve(self.X_train.t(),L_A,upper=False)[0]

        # predictive mean Xm depends on the term $X S_N X^T$, which we compute through a Cholesky decomposition
        Xm =  self.beta * H1T_star.t() @ H1T_train @ self.Y_train
        pred_mean = Xm # predictive mean

        # predictive covariance depends on the same terms
        XSX = H1T_star.t() @ H1T_star
        pred_covar = 1/self.beta*torch.eye(XSX.shape[0]).type(torch.double) + XSX
        return (pred_mean,pred_covar.diag())
        # Note that if you return a MultivariateNormal object, you'll have to perform Cholesky on a NxN matrix, which 
        # produces a cryptic memory error. You could avoid this by using GPyTorch's LazyTensor, but that's complicated


    def log_evidence(self, X, Y):
        # Em term
        # norm term
        A = self.alpha * torch.eye(self.n_features).type(torch.double) + self.beta) * X.t() @ X
        L_A = torch.cholesky(self.jitter(A))
        H1T = torch.triangular_solve(X.t(), L_A, upper=False)[0]
        norm = (Y.t() @ Y  +  self.beta**2 * Y.t() @ H1T.t() @ H1T @ H1T.t() @ H1T @ Y
                - 2 * self.beta @ Y.t() @ H1T.t() @ H1T @ Y)
        norm_term = self.beta / 2 * norm
        # mm term
        H2T = torch.triangular_solve(H1T, L_A.t(), upper=False)[0]
        mm = self.beta**2 * Y.t() @ H2T.t() @ H2T @ Y
        mm_term = self.alpha / 2 * mm
        # whole thing
        Em = norm_term + mm_term

        # logdet_tern
        logdet_term = L_A.trace()

        # whole thing
        log_evidence = (self.n_features / 2 * self.log_alpha + self.n_samples / 2 * self.log_beta - Em 
                        - logdet_term - self.n_samples / 2 * torch.log(2 * torch.Tensor([np.pi]).type(torch.double)))
        return log_evidence


    # Helper functions
    
    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def jitter(self, M):
        return M + self.jitter_factor * torch.eye(M.shape[0]).type(torch.double)
