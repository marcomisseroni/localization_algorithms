import numpy as np
import math

from core.base import Localizer
from typing import override

class BeamformingGccLs(Localizer):
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

    def __init__(self, sub_w_size: int, w_size: int, sensor_pos: np.ndarray, fs: int, c: float) -> None:
        super().__init__(sensor_pos, fs, c)
        self.Nw = w_size
        self.L = sub_w_size
        self.M = np.zeros((self.N, self.Nw))
        self.C = np.zeros((self.L * self.N, self.L * self.N))
        self.n_received = 0
        self.full = False
        self.tau_m = self.tau_max()


    @override
    def push(self, block: np.ndarray) -> np.ndarray | None:
        self.push_meas(block)

        if(not self.full):
            return None

        self.beamforming()
        # GCC
        # LS

    @override
    def reset(self) -> None:
        """
        Reset all the used matricies
        """
        self.M = np.zeros((self.N, self.Nw))
        self.C = np.zeros((self.L * self.N, self.L * self.N))
        self.n_received = 0
        self.full = False

    def tau_max(self) -> int:
        if(self.N <= 2):
            print("Error: not enough sebsors")
            return
        max_dist = 0
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                curr = math.dist(self.sensor_pos[i], self.sensor_pos[j])
                if(curr > max_dist):
                    max_dist = curr
        return max_dist * self.fs / self.c
    
    def build_X_mat(self, t: int, Mcp: np.ndarray) -> np.ndarray:
        """
        Build the X matrix from the measurements in M matrix. 
        The structure of X = [x1, x2, ..., xN].T
        Where xi = [xi(t), xi(t-1), ..., xi(t-L+1)].T
        So if t=0 we take the last L samples included the most recent one.
        """
        x = np.zeros((self.N * self.L))
        for i in range(self.N):
            x[i*self.L:(i+1)*self.L] = Mcp[i, t:t+self.L]
        return x

    def build_D_mat(self) -> np.ndarray:
        K = self.Nw - self.L + 1
        D = np.zeros((self.L * self.N, K))
        Mcp = self.M - self.M.mean(axis=1, keepdims=True)
        for i in range(K):
            X = self.build_X_mat(i, Mcp)
            D[:, i] = X
        return D
        

    def push_meas(self, block: np.ndarray) -> None:
        """
        Push the new measurement.
        """
        j = block.shape[1]
        self.n_received = min(self.n_received + j, self.Nw)
        self.full = self.n_received == self.Nw
        if(j >= self.Nw):
            block = block[:, j-self.Nw:]
            self.M = block[:, ::-1].copy()
            return
        self.M[:, j:] = self.M[:, 0:self.Nw-j]
        self.M[:, 0:j] = block[:, ::-1] # Invert the block since it arrives with new samples on the right 

    def corr_matrix_upd(self) -> None:
        """
        Compute the new correlation matrix.
        It uses the zero-mean signal
        """
        K = self.Nw - self.L + 1
        self.D = self.build_D_mat()
        self.C = self.D @ self.D.T / K

    def beamforming(self) -> np.ndarray:
        """
        Beamforming is a method used to increase the SNR by giving a different weight to
        each sensor.
        """
        self.corr_matrix_upd()
        w, v = np.linalg.eigh(self.C)
        v_max = v[:, -1]
        y = v_max.T @ self.D
        return y

    def generalized_cross_correlation(self, y: np.ndarray) -> None:
        """
        Use zero padding to expand the anchor sensor, then compute the cross correlation.
        """
        y = np.append(y, np.zeros(self.Nw - 1))
        y = np.concatenate([np.zeros(self.Nw - 1), y])
        for i in range(2 * self.tau_m + 1):
            pass

    def least_squares(self) -> np.ndarray:
        pass