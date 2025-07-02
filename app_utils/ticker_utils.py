import logging

DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "SBIN.NS", "INFY.NS",
    "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS", "AXISBANK.NS"
]

logger = logging.getLogger(__name__)

def load_tickers_from_file(file_path="tickers.txt"):
    """Load tickers from a text file, one ticker per line."""
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        logger.info(f"Ticker file {file_path} not found, using default tickers")
        return DEFAULT_TICKERS
    except Exception as e:
        logger.error(f"Error loading tickers from file: {e}")
        return DEFAULT_TICKERS

def save_tickers_to_file(tickers, file_path="tickers.txt"):
    """Save tickers to a text file, one ticker per line."""
    try:
        with open(file_path, 'w') as f:
            for ticker in tickers:
                f.write(f"{ticker.strip().upper()}\n")
        return True
    except Exception as e:
        logger.error(f"Error saving tickers to file: {e}")
        return False

def get_available_tickers():
    """Get list of available tickers. Loads from file if available, otherwise uses defaults."""
    return load_tickers_from_file()

def validate_ticker_format(ticker):
    """Validate ticker symbol format."""
    if not ticker:
        return False, "Ticker cannot be empty"
    
    ticker = ticker.strip()
    if len(ticker) < 1:
        return False, "Ticker too short"
    
    if len(ticker) > 20:
        return False, "Ticker too long (max 20 characters)"
    
    # Allow alphanumeric characters, dots, hyphens, and underscores
    allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_')
    if not all(c in allowed_chars for c in ticker.upper()):
        return False, "Invalid characters in ticker. Use only letters, numbers, dots, hyphens, and underscores"
    
    return True, "Valid ticker format"
