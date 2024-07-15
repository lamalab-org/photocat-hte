import sys
import glob
import serial
import serial.tools.list_ports
import time


def serial_ports():
    """Lists serial port names

    Raises:
        EnvironmentError: Unsupported platform

    Returns:
        list: List of serial ports
    """
    if sys.platform.startswith("win"):
        ports = ["COM%s" % (i + 1) for i in range(256)]
    elif sys.platform.startswith("linux") or sys.platform.startswith("cygwin"):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob("/dev/tty[A-Za-z]*")
    elif sys.platform.startswith("darwin"):
        ports = glob.glob("/dev/tty.*")
    else:
        raise EnvironmentError("Unsupported platform")

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


def check_com_port(port: str, name: str) -> bool:
    # the name is the port description that might be none
    # if not none, use it to double check if this matches
    # the port we are looking for
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if p.device == port:
            if name is None or name in p.description:
                return True
    return False


def find_com_port(name: str) -> str:
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if name in p.description:
            return p.device
    return None


def send_to_arduino(port, direction):
    ard = serial.Serial(port=port, baudrate=9600)
    time.sleep(2)
    ard.write(direction.encode("utf-8"))
    ard.close()


if __name__ == "__main__":
    print(serial_ports())
