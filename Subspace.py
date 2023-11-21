class StateSpaceModel:
    """
    Class for fitting and predicting a state space model using Hankel matrix decomposition
    """

    def __init__(self, k: int):
        """
        k (int): dimension of the state space model
        """
        self.is_fit = False
        self.k = k # hidden dimension

    def fit(self, X: pd.DataFrame):
        """
        Fits the state space model to the input data

        Parameters:
        X (pd.DataFrame): input data with rows as observations and columns as features

        Returns:
        self (StateSpaceModel): returns self for method chaining
        """

        # Convert input to np.ndarray if it's a pd.DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        self.is_fit = True
        self.m = X.shape[1] # the number of features
        self.n = X.shape[0] # the number of entries

        H = Hankel_matrix(X)

        Gamma, Omega = self.truncated_SVD(H)
        self.A, self.C, self.x0 = self.extract_variables(Gamma, Omega)
        self.CAn_1 = Gamma[-self.m:, :] # m * k matrix
        self.An_1x0 = Omega[:, -1] # k * 1 vector
        return self

    def truncated_SVD(self, H: np.ndarray):
        """
        Computes the truncated Singular Value Decomposition of the Hankel matrix

        Parameters:
        H (np.ndarray): Hankel matrix

        Returns:
        Gamma (np.ndarray): truncated Gamma matrix
        Omega (np.ndarray): truncated Omega matrix
        """
        U, S, VT = np.linalg.svd(H)
        if self.k > S.shape[0]:
            raise ValueError(f'The state space dimension {self.k} is too big. The data support k no more than {S.shape[0]}.')

        S_matrix = np.diag(np.sqrt(S[:self.k]))
        U_trunc = U[:, :self.k]
        VT_trunc = VT[:self.k, :]

        self.Gamma = U_trunc.dot(S_matrix)
        self.Omega = S_matrix.dot(VT_trunc)
        return self.Gamma, self.Omega

    def extract_variables(self, Gamma: np.ndarray, Omega: np.ndarray):
        """
        Extracts state space model variables from the truncated Gamma and Omega matrices

        Parameters:
        Gamma (np.ndarray): truncated Gamma matrix
        Omega (np.ndarray): truncated Omega matrix

        Returns:
        A (np.ndarray): state transition matrix
        C (np.ndarray): observation matrix
        x0 (np.ndarray): initial state vector
        """
        C = Gamma[:self.m, :] # m * k matrix
        segment_1 = Gamma[:-self.m, :] # ((n_lags - 1) * m) * k matrix
        segment_2 = Gamma[self.m:, :] # ((n_lags - 1) * m) * k matrix
        A = np.linalg.pinv(segment_1).dot(segment_2) # k * k matrix
        x0 = Omega[:, 0] # k * 1 vector
        if check_matrix_norm(A):
          warnings.warn(f'Matrix A has norm greater than one. This may upset further calculations (k={self.k}).', category=Warning)
        return A.copy(), C.copy(), x0.copy()

    def filtrate(self):
        """
        Approximate the output of the state space model for a trainig set

        Parameters:

        Returns:
        y (np.ndarray): filtered output for each time step in the training set
        """
        y = np.zeros((self.n, self.m))
        H = self.Gamma @ self.Omega
        nh = H.shape[1]
        nstr = H.shape[0] // self.m

        for j in range(nstr):
              y[j:j+nh,:] = y[j:j+nh,:] + H[self.m*j:self.m*(j+1),:].T
        for j in range(nh):
            y[j,:] = y[j,:]/(j+1)
            if j != (self.n - (j + 1)): 
                y[-j-1,:] = y[-j-1,:]/(j+1) 
        return y

    def predict(self, horizon: int):
        """
        Predicts the output of the state space model for a given horizon

        Parameters:
        horizon (int): number of future time steps to predict

        Returns:
        y (np.ndarray): predicted output for each time step in the horizon
        """
        y = np.zeros((horizon, self.m))
        z = self.An_1x0.copy() # the last instant at the training set

        for j in range(horizon):
            z = self.A @ z
            y[j,:] = self.CAn_1 @ z
        return y
    
    def periods(self):
        '''
        Finds oscillation periods
        
        Returns:
        freq (np.ndarray): calculated frequencies based on the eigenvalues of matrix A
        '''
        EV, S = np.linalg.eig(self.A)
        W = np.log(EV)
        freq = 2 * np.pi / np.abs(W.imag)
        # remove infty
        mask = np.isinf(freq)
        freq = freq[~mask][::2]
        return freq

def check_matrix_norm(A):
    l1_norm = np.linalg.norm(A, ord=1)
    l2_norm = np.linalg.norm(A, ord=2)
    linf_norm = np.linalg.norm(A, ord=np.inf)
    return (min(l1_norm, l2_norm, linf_norm) > 1)

def Hankel_matrix(X):
    """
    Creates a Hankel matrix from the input data.

     Args:
    - X (pd.DataFrame or np.ndarray): input data of shape (n_samples, n_features).

    Returns:
     - H (np.ndarray): Hankel matrix of shape (n_lags*n_features, n_samples-n_lags+1,), where n_lags = n_samples // 2.
    """

     # Convert input to np.ndarray if it's a pd.DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Determine number of lags
    n_lags = X.shape[0] // 2 + 1

    # Create Hankel matrix
    H = np.zeros((n_lags * X.shape[1], X.shape[0] - n_lags + 1))
    for i in range(H.shape[1]):
         H[:, i] = X[i:i+n_lags, :].reshape(-1)

    return H

def MAE(y, y_pred, cumulative=False):
  u = np.abs((y - y_pred))
  if cumulative:
    v = np.ones(u.shape)
    return u.cumsum(axis=0) / v.cumsum(axis=0)
  else:
    return u.mean(axis=0)