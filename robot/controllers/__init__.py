__all__ = ["controller", "GenesisRobotController"]


def __getattr__(name):
    if name == "controller":
        from .robot import controller

        return controller

    if name == "GenesisRobotController":
        from .genesis_robot import GenesisRobotController

        return GenesisRobotController

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
