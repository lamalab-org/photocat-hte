from pyrolib import PyroDevice
import io
from contextlib import redirect_stdout, redirect_stderr


def measure_firesting(port: str) -> dict:
    """
    Measures the firesting on the specified port.

    Args:
        port (str): The port to which the firesting is connected.
        For example: "/dev/ttyUSB0" or "COM3"

    Returns:
        dict: A dictionary containing the measured values.
    """
    # we redirect the output to a string buffer
    # because stdout prints warnings that we don't care about
    with redirect_stderr(io.StringIO()) as _stderr:
        with redirect_stdout(io.StringIO()) as _stdout:
            firesting = PyroDevice(port)
            result = firesting.measure()
            all_results = {}
            for channel, data in result.items():
                for k, v in data.items():
                    all_results[k + "_" + str(channel)] = v
            return all_results
