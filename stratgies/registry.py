from .ema_rsi import EMARSI


STRATEGY_REGISTRY = {"EMARSI": EMARSI}


def get_strategy(strategy_name: str):
    return STRATEGY_REGISTRY[strategy_name]
