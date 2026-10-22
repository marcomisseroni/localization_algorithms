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
        self.tau_m = int(self.tau_max())
        if(self.tau_m > self.L - 1):
            raise ValueError("The current sub-window is not large enough to capture the same event accross all sensor pairs")
        if(self.L > self.Nw):
            raise ValueError("The sub-window must be smaller than the data window")
        if(self.N < 4):
            raise ValueError("For the solution of least squares at least 4 sensors are required")

    @override
    def push(self, block: np.ndarray) -> np.ndarray | None:
        self.push_meas(block)

        if(not self.full):
            return None

        y = self.beamforming()
        cc = np.zeros((self.N, self.L))
        tau = np.zeros(self.N)
        for i in range(self.N):
            cc[i, :] = self.cross_correlation(y, self.M[i, :])
            tau[i] = np.argmax(cc[i, :])
        dtau = (tau[0] - tau) / self.fs
        theta = self.least_squares(dtau)
        return theta[:2].ravel()

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
        k = np.argmax(np.abs(v_max))
        if(v_max[k] < 0):
            v_max = -v_max
        y = v_max.T @ self.D
        return y

    def cross_correlation(self, y: np.ndarray, x:np.ndarray) -> np.ndarray:
        """
        Use zero padding to expand x signal, then compute the cross correlation.
        To have x and y aligned with tau = 0 we have to overlap the first element of x with the first 
        element of y.
        """
        x = x - x.mean()
        cc = np.zeros((1, self.L))
        for i in range(self.L):
            cc[0, i] = y @ x[i:i+y.shape[0]]
        return cc

    def least_squares(self, tau: np.ndarray) -> np.ndarray:
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
        return np.linalg.lstsq(A, b, rcond=None)[0]

if __name__ == "__main__":
    # Footstep-like test: 3 sensors on the floor ~1 m apart, slow floor wave
    fs = 1000                                           # [Hz]
    c = 300                                             # [m/s]
    sensor_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    source_pos = np.array([0.7, 0.4, 0])

    # tau_max = sqrt(2) * fs / c ~ 4.7 samples -> L - 1 = 7 is enough
    # K = Nw - L + 1 = 505 columns to estimate a (N*L x N*L) = 24 x 24 matrix
    loc = BeamformingGccLs(8, 512, sensor_pos, fs, c)

    # Broadband low-frequency burst (low-passed noise under a Hann window), like a footstep
    rng = np.random.default_rng(0)
    T = 1024
    t = np.arange(T)
    noise = np.convolve(rng.standard_normal(T), np.ones(9) / 9, mode="same")
    burst = np.zeros(T)
    burst[400:600] = noise[400:600] * np.hanning(200)

    # Fractional delays from the geometry, relative to the first sensor reached
    delays = np.linalg.norm(sensor_pos - source_pos, axis=1) / c * fs
    delays -= delays.min()
    print("true delays [samples]:", np.round(delays, 2))
    signals = np.stack([np.interp(t - d, t, burst) for d in delays])
    signals += 0.01 * rng.standard_normal(signals.shape)

    # Simulate the real-time stream
    hop = 64
    for k in range(0, T, hop):
        loc.push(signals[:, k:k + hop])
