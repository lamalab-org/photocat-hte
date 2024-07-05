from rd6006 import RD6006
import serial
import time
from photolooper.utils import send_to_arduino

def change_power(port: str, voltage: float):
    ps = RD6006(port)
    ps.voltage = voltage
    ps.enable


def switch_on(port: str, arduino_port: str = 'COM7', voltage: float = 0.18):
    send_to_arduino(arduino_port, 'ON')
    return change_power(port, voltage)


def switch_off(port: str, arduino_port: str = "COM7"):
    """
    Switches off the power supply connected to the specified port.

    Args:
        port (str): The port to which the power supply is connected.

    Returns:
        None
    """
    # Switch the power supply off by setting the voltage to 0
    send_to_arduino(arduino_port, 'OFF')
    change_power(port, 0)
