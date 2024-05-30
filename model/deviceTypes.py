from enum import Enum

class DeviceTypes(str, Enum):
    cpu = "cpu"
    gpu = "gpu"
    