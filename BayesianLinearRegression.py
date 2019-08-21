import numpy as np
import torch
import torch.nn as nn
import gpytorch
from my_ml_utils import rmse


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

        A = self.alpha() * torch.eye(self.n_features).type(torch.double) + self.beta() * self.X_train.t() @ self.X_train
        L_A = torch.cholesky(self.jitter(A))
        H1T_star = torch.triangular_solve(X.t(), L_A, upper=False)[0] # Have to compute the transpose because of the way triangular_solve/trtrs is written
        H1T_train = torch.triangular_solve(self.X_train.t(),L_A,upper=False)[0]

        # predictive mean Xm depends on the term $X S_N X^T$, which we compute through a Cholesky decomposition
        Xm =  self.beta() * H1T_star.t() @ H1T_train @ self.Y_train
        pred_mean = Xm # predictive mean

        # predictive covariance depends on the same terms
        XSX = H1T_star.t() @ H1T_star
        pred_covar = 1/self.beta()*torch.eye(XSX.shape[0]).type(torch.double) + XSX
        return (pred_mean,pred_covar.diag())
        # Note that if you return a MultivariateNormal object, you'll have to perform Cholesky on a NxN matrix, which 
        # produces a cryptic memory error. You could avoid this by using GPyTorch's LazyTensor, but that's complicated


    def log_evidence(self, X, Y):

        # Em term
        # norm term
        A = self.alpha() * torch.eye(self.n_features).type(torch.double) + self.beta() * X.t() @ X
        L_A = torch.cholesky(self.jitter(A))
        H1T = torch.triangular_solve(X.t(), L_A, upper=False)[0]
        norm = (Y.t() @ Y  +  self.beta()**2 * Y.t() @ H1T.t() @ H1T @ H1T.t() @ H1T @ Y
                - 2 * self.beta() @ Y.t() @ H1T.t() @ H1T @ Y)
        norm_term = self.beta() / 2 * norm
        # mm term
        H2T = torch.triangular_solve(H1T, L_A.t(), upper=False)[0]
        mm = self.beta()**2 * Y.t() @ H2T.t() @ H2T @ Y
        mm_term = self.alpha() / 2 * mm
        # whole thing
        Em = norm_term + mm_term

        # logdet_tern
        logdet_term = L_A.trace()

        # whole thing
        log_evidence = (self.n_features / 2 * self.log_alpha + self.n_samples / 2 * self.log_beta - Em 
                        - logdet_term - self.n_samples / 2 * torch.log(2 * torch.Tensor([np.pi]).type(torch.double)))
        return log_evidence


    # Helper functions
    
    def alpha(self):
        return torch.exp(self.log_alpha)

    def beta(self):
        return torch.exp(self.log_beta)

    def jitter(self, M):
        return M + self.jitter_factor * torch.eye(M.shape[0]).type(torch.double)



def train_lbfgs(model, X_train=None, Y_train=None, X_test=None, Y_test=None, training_iterations=50, verbose=False, lr=0.5):

    if ((type(X_train) is type(None)) | (type(Y_train) is type(None)) | (type(X_test) is type(None)) | (type(Y_test) is type(None))):
        raise ValueError('Need to set train and test matrices')

    optimizer = optim.LBFGS(model.parameters(), lr=lr)
    def closure():
        optimizer.zero_grad()
        loss = -model.log_evidence(X_train,Y_train)
        loss.backward()
        return loss

    for i in range(training_iterations):
        optimizer.step(closure)
        if verbose:
            if i % 10 == 0:
                loss = -model.log_evidence(X_train,Y_train)
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                print(model.alpha(),model.beta())
            if i % 10 == 0:
                preds_test = model(X_test)
                preds_train = model(X_train)

                P_test = preds_test[0]
                P_train = preds_train[0]
                print('### RMSE train: %.1f  RMSE test: %.1f ###' % (
                rmse(Y_train.data.numpy(),P_train.data.numpy()),rmse(Y_test.data.numpy(),P_test.data.numpy())
                ))



if __name__ == '__main__':
    import os
    import pandas as pd
    import numpy as np
    import torch.optim as optim
    from diabetes_utils import convert_uk_to_eu, plot_clarke
    from my_ml_utils import rmse
    from LBFGS import FullBatchLBFGS
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures


    ######### PATIENT AND TIME ##########
    patient_name = 'CAM11-020-0007-SAP'
    time_idx = 3
    poly_degree = 1
    #####################################


    # Loading data and preprocessing

    roation_path = os.environ['ROATION_WITH_RASMUSSEN_ROMAN']

    X_train = pd.read_csv(roation_path + '/data/X_and_Y_future_no_leakage/' + patient_name + '_X_train.tsv',sep='\t').set_index('time')
    Y_train_times = pd.read_csv(roation_path + '/data/X_and_Y_future_no_leakage/' + patient_name + '_Y_train.tsv',sep='\t').set_index('time')
    X_test = pd.read_csv(roation_path + '/data/X_and_Y_future_no_leakage/' + patient_name + '_X_test.tsv',sep='\t').set_index('time')
    Y_test_times = pd.read_csv(roation_path + '/data/X_and_Y_future_no_leakage/' + patient_name + '_Y_test.tsv',sep='\t').set_index('time')

    mask_past = ~X_train.columns.str.contains('plus')
    X_train = X_train.loc[:,mask_past]
    X_test = X_test.loc[:,mask_past]

    poly = PolynomialFeatures(poly_degree)
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)

    Y_train_times = convert_uk_to_eu(Y_train_times)
    Y_test_times = convert_uk_to_eu(Y_test_times)

    X_train = torch.Tensor(X_train).type(torch.double)
    Y_train_times = torch.Tensor(Y_train_times.values).type(torch.double)
    X_test = torch.Tensor(X_test).type(torch.double)
    Y_test_times = torch.Tensor(Y_test_times.values).type(torch.double)

    Y_train = Y_train_times[:,time_idx:time_idx+1]
    Y_test = Y_test_times[:,time_idx:time_idx+1]

    # Define model and train

    model = BayesianLinearRegression(X_train,Y_train)


             

    def train_adam(training_iterations=500):

        optimizer = optim.Adam(model.parameters(), lr=0.1)

        optimizer.zero_grad()
        loss = -model.log_evidence(X_train, Y_train)
        loss.backward()

        for i in range(training_iterations):
            optimizer.zero_grad()
            loss = -model.log_evidence(X_train, Y_train)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                print(model.alpha(),model.beta())
            if i % 100 == 0:
                preds_test = model(X_test)
                preds_train = model(X_train)

                P_test = preds_test[0]
                P_train = preds_train[0]
                print('### RMSE train: %.1f  RMSE test: %.1f ###' % (
                rmse(Y_train.data.numpy(),P_train.data.numpy()),rmse(Y_test.data.numpy(),P_test.data.numpy())
                ))        

    train_lbfgs(modelX_train,Y_train)

    P_test = model(X_test)[0]
    P_train = model(X_train)[0]

    plt.scatter(Y_test.data.numpy(),P_test.data.numpy())
    plt.title('Test')
    plt.text(0.8, 0.2, 'RMSE %.1f mg/dL' % rmse(Y_test.data.numpy(),P_test.data.numpy()),transform = plt.gca().transAxes)
    plt.show()
    plt.scatter(Y_train.data.numpy(),P_train.data.numpy())
    plt.title('Train')
    plt.text(0.8, 0.2, 'RMSE %.1f mg/dL' % rmse(Y_train.data.numpy(),P_train.data.numpy()),transform = plt.gca().transAxes)
    plt.show()

    import pdb; pdb.set_trace()

