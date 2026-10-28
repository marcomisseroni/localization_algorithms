import numpy as np
import math

from localization.base import Localizer
from typing import override
from scipy.signal import correlate
from numpy.fft import fft, ifft

class BeamformingGccLs(Localizer):
    """
    Implementation of a localization algorithm, beamforming to increare SNR + 
    generalized cross-correlation + least squares.
    Nw: Number of samples (the windows must be greater than the max possible delay)
    N: Number of sensors
    L: Length of the sub window
    X: Matrix containing L samples for each sensor, X(t) contains L samples and the last one is collected now. DIM = (NL * 1)
    C: Correlation matrix DIM = (NL * NL)
    M: Measurement matrix, last Nw samples of each sensor, each row is a different sensor, samples on cols
    
    Warning: before using this algorithm on measurements of vibration of the ground a band pass filter should be applied 
    in order to extract a small frequency interval and the speed is almost constant.
    """

    def __init__(self, L: int, Nw: int, sensor_pos: np.ndarray, fs: int, c: float, room_dim: list[float], cc_type: str, pre_margin: int) -> None:
        super().__init__(sensor_pos=sensor_pos, fs=fs, c=c, pre_margin=pre_margin, room_dim=room_dim)
        self.Nw = Nw
        self.L = L
        self.M = np.zeros((self.N, self.Nw))
        self.C = np.zeros((self.L * self.N, self.L * self.N))
        self.D = None
        self.n_received = 0
        self.full = False
        self.tau_m = int(self.tau_max())
        self.cc_type = cc_type  # possible options: cc, roth, scot, phat
        if(self.tau_m > self.L - 1):
            raise ValueError("The current sub-window is not large enough to capture the same event accross all sensor pairs")
        if(self.L > self.Nw):
            raise ValueError("The sub-window must be smaller than the data window")
        if(self.N < 4):
            raise ValueError("For the solution of least squares at least 4 sensors are required")

    @override
    def reset(self) -> None:
        """
        Reset all the used matricies
        """
        self.M = np.zeros((self.N, self.Nw))
        self.C = np.zeros((self.L * self.N, self.L * self.N))
        self.D = None
        self.n_received = 0
        self.full = False

    @override
    def load_block_from_det(self, block: np.ndarray) -> np.ndarray | None:
        n_rows = block.shape[0]
        n_cols = block.shape[1]
        if((self.L > n_cols) or (self.N != n_rows)):
            print("Wrong dimension of the block")
            return None
        self.Nw = n_cols
        self.M = block[:, ::-1].copy()
        self.full = True
        return self.estimate()

    @override
    def estimate(self) -> np.ndarray | None:
        if(not self.full):
            return None
        y = self.beamforming()
        cc = np.zeros((self.N, len(y) + self.Nw - 1))
        tau = np.zeros(self.N)
        for i in range(self.N):
            cc[i, :] = self.generalized_cross_correlation(y, self.M[i, :])
            tau[i] = np.argmax(cc[i, :]) - (len(y) - 1)
        dtau = (tau[0] - tau) / self.fs
        theta = self.least_squares(dtau)
        if(theta is None):
            return None
        return theta[:2].ravel()

    def tau_max(self) -> float:
        """
        Find the distance between any sensor pair to compute the max physical delay between 2 sensors.
        """
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

    def corr_matrix_upd(self) -> None:
        """
        Compute the new correlation matrix.
        It uses the zero-mean signal
        """
        K = self.Nw - self.L + 1
        self.D = self.build_D_mat()
        self.C = self.D @ self.D.T / K

    def beamforming(self) -> np.ndarray:    # Check SNR for x and y
        """
        Beamforming is a method used to increase the SNR by giving a different weight to
        each sensor.
        """
        self.corr_matrix_upd()
        w, v = np.linalg.eigh(self.C)
        v_max = v[:, -1]
        k = np.argmax(np.abs(v_max))
        if(v_max[k] < 0):
            v_max = -v_max
        y = v_max.T @ self.D
        return y

    def cross_correlation(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute the cross correlation via simple multiplication.
        """
        cc = correlate(x - x.mean(), y, mode="full")
        return cc

    def generalized_cross_correlation(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute the generalized cross-correlation, steeper peak so more precise time lag estimation.
        """
        cc = correlate(x - x.mean(), y, mode="full")
        window = None
        match self.cc_type:
            case "cc":
                return cc
            case "roth":
                Ry = correlate(y, y, mode="full")
                n = max(len(cc), len(Ry))
                Py = np.abs(fft(Ry, n=n))
                epsilon = max(1e-3 * np.max(np.abs(Py)), 1e-12)
                window = (Py + epsilon)**-1
            case "scot":
                Rx = correlate(x - x.mean(), x - x.mean(), mode="full")
                Ry = correlate(y, y, mode="full")
                n = max(len(cc), len(Ry), len(Rx))
                Px = np.abs(fft(Rx, n=n))
                Py = np.abs(fft(Ry, n=n))
                epsilon = max(1e-3 * np.sqrt(Px * Py).max(), 1e-12) 
                window = 1 / (np.sqrt(Px * Py) + epsilon)
            case "phat":
                n = len(cc)
                Pyx = np.abs(fft(cc, n=n))
                epsilon = max(1e-3 * np.max(np.abs(Pyx)), 1e-12)
                window = (Pyx + epsilon)**-1
        csd = fft(cc, n=n)
        return ifft(csd * window).real[:len(cc)]

    def least_squares(self, tau: np.ndarray) -> np.ndarray | None:
        """
        Solve the system A@theta = b.
        Where: 
        A = [[2*dx2, 2*dy2, 2*d21], [2*dx3, 2*dy3, 2*d31], ..., [2*dxn, 2*dyn, 2*dn1]]
        b = [K2 - K1 - d12^2, K3 - K1 - d31^2, ..., Kn - K1 - dn1^2].T
        theta = [Px, Py, r1].T
        dxi = xi - x1
        dyi = yi - y1
        di1 = c * taui1
        Ki = ||Si||**2
        P = [Px, Py] is the source position
        Si = [xi, yi] is the i-th sensor's position
        r1 = ||P - S1||
        So we can calculate the least squares with: theta = (A.T @ A).inv @ A.T @ b
        """
        A = np.zeros((self.N - 1, 3))
        b = np.zeros((self.N - 1, 1))
        x1 = self.sensor_pos[0, 0]
        y1 = self.sensor_pos[0, 1]
        for i in range(self.N - 1):
            xi = self.sensor_pos[i + 1, 0]
            yi = self.sensor_pos[i + 1, 1]
            di1 = self.c * tau[i + 1]
            A[i, 0] = 2 * (xi - x1)
            A[i, 1] = 2 * (yi - y1)
            A[i, 2] = 2 * di1
            b[i] = xi**2 + yi**2 - x1**2 - y1**2 - di1**2
        theta_opt = np.linalg.lstsq(A, b, rcond=None)[0]
        d1s = np.linalg.norm(self.sensor_pos[0, :2] - np.array([theta_opt[0, 0], theta_opt[1, 0]]))
        if(theta_opt[2] < 0 or np.abs(d1s - theta_opt[2, 0]) > 0.1 * theta_opt[2, 0]):
            theta_opt = None
        return theta_opt