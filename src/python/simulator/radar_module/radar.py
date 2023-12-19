from python.simulator.radar_module.receiver import Receiver
from python.simulator.radar_module.transmitter import Transmitter


class Radar:
    """control the radar parameters"""

    def __init__(self, transmitter: Transmitter, receiver: Receiver) -> None:
        self.transmitter = transmitter
        self.receiver = receiver

    def transmit(self) -> np.ndarray:
        """ "transmit the band pass signal"""
        return self.transmitter.transmit()

    def receive(self, signal: np.ndarray) -> np.ndarray:
        """receive the signal band pass signal and convert him to base band"""
        return self.receiver.receive(signal)
