def get_features_by_indicator(timeframe, indicator_name):
    base_features = ["close", "high", "low", "volume"]
    strategy_features = {
        "5m": {
            "Trend-Following": {"features": base_features + ["ema_12", "ema_26", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_12", "ema_26", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        },
        "1h": {
            "Trend-Following": {"features": base_features + ["ema_20", "ema_50", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_20", "ema_50", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        },
        "4h": {
            "Trend-Following": {"features": base_features + ["ema_50", "ema_200", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_50", "ema_200", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        },
        "1d": {
            "Trend-Following": {"features": base_features + ["ema_50", "ema_200", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_50", "ema_200", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        }
    }

    # Check if the timeframe exists
    if timeframe not in strategy_features:
        return None

    # Check if the indicator_name exists in the timeframe
    if indicator_name not in strategy_features[timeframe]:
        return None

    # Return the features for the specified indicator
    return strategy_features[timeframe][indicator_name]["features"]


def get_strategy_name(timeframe, features):
    base_features = ["close", "high", "low", "volume"]
    strategy_features = {
        "5m": {
            "Trend-Following": {"features": base_features + ["ema_12", "ema_26", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_12", "ema_26", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        },
        "1h": {
            "Trend-Following": {"features": base_features + ["ema_20", "ema_50", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_20", "ema_50", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        },
        "4h": {
            "Trend-Following": {"features": base_features + ["ema_50", "ema_200", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_50", "ema_200", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        },
        "1d": {
            "Trend-Following": {"features": base_features + ["ema_50", "ema_200", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_50", "ema_200", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        }
    }

    # Check if the timeframe exists in the strategy_features
    if timeframe in strategy_features:
        # Iterate through the strategies in the given timeframe
        for strategy_name, strategy_data in strategy_features[timeframe].items():
            # Check if the features match the strategy's feature list
            if sorted(features) == sorted(strategy_data["features"]):
                return strategy_name

    # If no match is found, return None or a default value
    return None