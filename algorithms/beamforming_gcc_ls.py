import numpy as np


class SourceLocalizer:
    """
    Implementation of a localization algorithm, beamforming to increare SNR + 
    generalized cross-correlation + least squares
    Nw: Number of samples (the windows must be greater than the max possible delay)
    N: Number of sensors
    L: Length of the sub window
    X: Matrix containing L samples for each sensor, X(t) contains L samples and the last one is collected now. DIM = (NL * 1)
    C: Correlation matrix DIM = (NL * NL)
    M: Measurement matrix, last Nw samples of each sensor, each row is a different sensor, samples on cols
    """

    Nw = None
    N = None
    L = None
    X = None
    C = None
    M = None

    def __init__(self, sub_w_size: int, n_sensors: int, w_size: int) -> None:
        self.Nw = w_size
        self.N = n_sensors
        self.L = sub_w_size
        self.M = np.zeros((self.N, self.Nw))
        self.X = np.zeros((self.N * self.L, 1))
        self.C = np.zeros((self.L * self.N, self.L * self.N))

    def build_X_mat(self, t: int) -> np.ndarray:
        """
        Build the X matrix from the measurements in M matrix. 
        The structure of X = [x1, x2, ..., xN].T
        Where xi = [xi(t), xi(t-1), ..., xi(t-L+1)].T
        So if t=0 we take the last L samples included the most recent one.
        """
        x = np.zeros((self.N * self.L))
        M_rows, M_cols = self.M.shape
        for i in range(self.N):
            x[i*self.L:(i+1)*self.L] = self.M[i, t:t+self.L].T

        return x

    def corr_matrix_upd(self):
        K = self.Nw - self.L + 1
        D = np.zeros((self.L * self.N, K))
        for i in range(K):
            X = self.build_X_mat(i)
            D[:, i] = X
        self.C = D @ D.T / K

    def push_meas(self):
        # Push the new meas for each sensor
        # Subtract the mean of the window from each sensor
        pass

def main():
    SL = SourceLocalizer(3, 2, 5)
    SL.M = np.zeros((2, 5))
    SL.M[0, :] = np.array([1, 2, 3, 4, 5])
    SL.M[1, :] = np.array([6, 7, 8, 9, 10])
    SL.corr_matrix_upd()
    print(SL.build_X_mat(1))
    print(SL.C)

if __name__ == "__main__":
    main()