import numpy as np
import pywt
import math
import shapely

from localization.base import Localizer
from typing import override
from scipy.spatial.distance import pdist
from shapely.geometry import LineString, box
from itertools import product

class WaveletTdoaMult(Localizer):
    """
    Wavelet to compute TDoA and adaptive multilateration to find the source position.

    M: matrix of measurements.
    freq: vector of frequencies in [low_cutoff, high_cutoff] used to compute the vector of scales for the wavelet transform.
    edge_margin: margin to exclude from wavelet transform to avoid peak in wrong frequencies due to 
    the transition from 0 to noise because of the zero padding in the CWT calculation.
    v_min, v_max, v_count: used to compute a vector of possible velocities to use in adaptive multilateration.
    h_range, h_steps: to compute a vector of h used to parametrize the hyperbolas
    k: used in the threshold for detecting an event from the wavelet matrix.
    room_box: to discard hyperbolas intersections outside of the room.
    """
    def __init__(self, sensor_pos: np.ndarray, fs: int, c: float | None, room_dim: list[float], 
                 pre_margin: int, low_cutoff: float, high_cutoff: float, n_scales: int,
                 v_min: float, v_max: float, v_count: int, h_range: float, h_steps: int, k: float) -> None:
        super().__init__(sensor_pos=sensor_pos, fs=fs, c=c, pre_margin=pre_margin, room_dim=room_dim, 
                         low_cutoff=low_cutoff, high_cutoff=high_cutoff)
        self.M: np.ndarray | None = None
        self.freq = np.geomspace(low_cutoff, high_cutoff, n_scales)
        self.scales = pywt.central_frequency('mexh') * self.fs / self.freq
        self.edge_margin = math.ceil(self.scales.max() * 5)    # depends only on the scales, so it is fixed
        self.v_min = v_min
        self.v_max = v_max
        self.v_count = v_count
        self.v_vec = np.linspace(self.v_min, self.v_max, self.v_count)
        self.h_vec = np.linspace(-h_range, h_range, h_steps)
        self.k = k
        self.room_box = box(0, 0, self.room_dim[0], self.room_dim[1])

    @override
    def load_block_from_det(self, block: np.ndarray) -> np.ndarray | None:
        self.M = block.copy()   # chronological order: the first samples are the noise before the event
        return self.estimate()

    @override
    def reset(self) -> None:
        self.M = None

    @override
    def estimate(self) -> np.ndarray | None:
        """
        Compute the wavelet, choose a threshold based on the variance of the noise or peak of wavelet times k.
        Build toa taking the first sample in which the wavelet is bigger than the threshold for each sensor for each frequency.
        It is build sorting the sensors with ToA in ascending order, so earlier detection first.
        Then call adaptive multilateration to find the source position.
        """
        if(self.M is None):
            return None
        coefs, f = self.wavelet(self.M)
        sigmas = np.std(coefs[:, :, 0:self.pre_margin], axis=-1)
        start = max(self.pre_margin, self.edge_margin)
        stop = coefs.shape[-1] - self.edge_margin
        if(start > stop):
            print("Error: pre_margin or edge_margin are larger than the block dimension")
            return None
        win = np.abs(coefs[:, :, start:stop])
        peak = win.max(axis=-1)
        thr = np.maximum(3 * sigmas, self.k * peak)
        mask = np.abs(coefs[:, :, start:stop]) > 3 * thr[:, :, None]
        toa_mat = np.full((len(self.freq), self.N), np.inf)
        for i in range(len(self.freq)):
            for j in range(self.N):
                curr = np.argmax(mask[i, j, :])
                if(curr or mask[i, j, 0]):
                    toa_mat[i, j] = curr

        toa_vec = toa_mat.min(axis=0)
        idx = np.argsort(toa_vec)
        asc_toa_mat = toa_mat[:, idx[0:4]]
        sensor_pos = self.sensor_pos[idx[0:4], :]
        tau_mat = (asc_toa_mat - asc_toa_mat[:, [0]]) / self.fs
        return self.adaptive_multilateration(tau_mat[:, 1:], sensor_pos)

    def wavelet(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute wavelet transform from the given signals.
        Coefs is S x N x L 
        where S is the scale axis, N is the sensor index and L is sample index.
        """
        coefs, f = pywt.cwt(data, self.scales, 'mexh', sampling_period=1/self.fs, method='fft', axis=-1)
        return coefs, f

    def adaptive_multilateration(self, tau_mat: np.ndarray, sensor_pos: np.ndarray) -> np.ndarray | None:
        """
        Iterates over each scale and each velocity in the velocity vector. For each combination computes the 
        intersections between hyperbolas and the relative cost. Choose the velocity that has minimum cost and take as 
        estimated position for that v the center of the 3 intersections. Then take the 8 positions with min cost over all 
        scales and return the average.
        """
        intersections = np.zeros((3, 2, len(self.v_vec)))
        J = np.zeros((len(self.freq), len(self.v_vec)))
        J_v_opt = np.zeros(len(self.freq))
        source_pos = np.zeros((len(self.freq), 2))
        for s in range(len(self.freq)):
            for v in range(len(self.v_vec)):
                intersections[:, :, v] = self.find_hyp_int(tau_mat[s, :], self.v_vec[v], sensor_pos) # Check if keep None as out of the func or change it
                J[s, v] = self.compute_J(intersections[:, :, v])
            v_opt = np.argmin(J[s, :])
            J_v_opt[s] = np.min(J[s, :])
            source_pos[s, :] = intersections[:, :, v_opt].mean(axis=0)
        idx = np.argsort(J_v_opt)
        idx = idx[np.isfinite(J_v_opt[idx])]
        if(len(idx) == 0):
            return None  
        best_pos = source_pos[idx[:8]]
        return best_pos.mean(axis=0)
        
    def find_hyp_int(self, tau_vec: np.ndarray, v: float, sensor_pos: np.ndarray) -> np.ndarray:
        """
        Build one hyperbola between the anchor sensor (first in sensor_pos) and each other sensor, using the TDoA
        in tau_vec and the velocity v. Then compute the intersections for each pair of hyperbolas, keeping only
        the ones inside the room as candidates (a pair can intersect in 2 points sharing the anchor as focus).
        Among all the combinations with one candidate per pair, return the 3 points with minimum cost.
        If a TDoA is not finite, a hyperbola does not exist (|a| >= c) or a pair has no intersection in the room,
        return the intersections filled with inf.
        """
        hyperbolas = []
        int_count = int(math.factorial(len(tau_vec)) / (2 * math.factorial(len(tau_vec) - 2)))
        intersections = np.full((int_count, 2), np.inf)
        if(not np.isfinite(tau_vec).all()):
            return intersections
        for i in range(len(tau_vec)):
            m = (sensor_pos[0, :2] + sensor_pos[i + 1, :2]) / 2
            c = np.linalg.norm(sensor_pos[i + 1, :2] - sensor_pos[0, :2]) / 2
            a = v * tau_vec[i] / 2
            if(abs(a) >= c):
                return intersections
            b = np.sqrt(c**2 - a**2)
            u1 = (sensor_pos[0, :2] - sensor_pos[i + 1, :2]) / np.linalg.norm(sensor_pos[0, :2] - sensor_pos[i + 1, :2])
            u2 = np.array([-u1[1], u1[0]])
            x = a * np.cosh(self.h_vec)
            y = b * np.sinh(self.h_vec)
            hyperbolas.append(m + x[:, None] * u1 + y[:, None] * u2)

        candidate_points = []
        for i in range(len(hyperbolas)):
            for j in range(i + 1, len(hyperbolas)):
                line_i = LineString(hyperbolas[i])
                line_j = LineString(hyperbolas[j])
                inter = line_i.intersection(line_j)
                if(inter.geom_type == "Point"):
                    points = [inter]
                elif(inter.geom_type == "MultiPoint"):
                    points = list(inter.geoms)
                else:
                    points = []
                points = [p for p in points if self.room_box.contains(p)]
                if(len(points) == 0):
                    return intersections
                candidate_points.append(points)

        comb_counts = list(product(*candidate_points))
        combinations = np.zeros((len(comb_counts), 3, 2))
        J = np.zeros(len(comb_counts))
        for i, p in enumerate(comb_counts):
            pts = np.array([[pt.x, pt.y] for pt in p])
            J[i] = self.compute_J(pts)
            combinations[i, :, :] = pts
        opt_points = J.argmin()
        return combinations[opt_points, :, :]

    def compute_J(self, intersections: np.ndarray) -> float:
        """
        Compute the cost as the sum of the norm of the difference between each pair of points.
        """
        cost = pdist(intersections).sum()
        if(np.isnan(cost)):
            cost = np.inf
        return cost