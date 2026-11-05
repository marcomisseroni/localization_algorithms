"""
Simulation of the datasets, used as a black box:

    from sim import simulate
    simulate("data/general/y_7m_center_snr20", sensor_pos, source_pos, fs=16000, snr_db=20)

writes in the folder dataset.json (common information and list of the recordings) and one multichannel
float32 WAV per source position. See sim/simulate.py for all the parameters.
"""

from sim.simulate import simulate
