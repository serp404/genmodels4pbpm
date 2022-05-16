import typing as tp

class BaseConfig:
    def __init__(self, params: tp.Dict[str, tp.Any]) -> None:
        for key, value in params.items():
            setattr(self, key, value)
