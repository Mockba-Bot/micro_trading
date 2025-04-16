from app.models.backtest import run_backtest

async def execute_backtest(
    pair: str,
    timeframe: str,
    token: str,
    values: str,
    stop_loss_threshold: float,
    free_collateral: float,
    maker_fee: float,
    taker_fee: float,
    gain_threshold: float,
) -> str:
    """
    Executes the backtest by calling the backtest logic from `backtest.py`.

    Args:
        pair (str): Trading pair (e.g., "BTCUSDT").
        timeframe (str): Time interval for the backtest (e.g., "1h").
        token (str): User token for API authentication.
        values (str): Date range for historical data (e.g., "2023-01-01|2023-12-31").
        stop_loss_threshold (float): Stop-loss percentage threshold.
        free_collateral (float): Initial investment for the backtest.
        maker_fee (float): Maker fee percentage.
        taker_fee (float): Taker fee percentage.
        gain_threshold (float): Gain percentage threshold.

    Returns:
        str: Backtest results explanation.
    """
    # Ensure `run_backtest` is async
    return await run_backtest(
        pair,
        timeframe,
        token,
        values,
        stop_loss_threshold,
        free_collateral,
        maker_fee,
        taker_fee,
        gain_threshold,
    )
