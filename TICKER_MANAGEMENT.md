# Dynamic Ticker Management

The backtesting application now supports dynamic ticker management, removing hardcoded ticker symbols and providing flexible ticker configuration options.

## Features

### 1. Dynamic Ticker Loading
- **Default Tickers**: Predefined list of popular Indian stock tickers
- **External File**: Load tickers from `tickers.txt` file (one ticker per line)
- **Fallback**: Uses default tickers if external file is not found

### 2. Ticker Input Methods
- **Select from List**: Choose from available tickers in dropdown
- **Custom Input**: Enter any ticker symbol manually with validation

### 3. Ticker Management UI
Access through the "Ticker Management" expander in the sidebar:

#### Add Tickers
- Enter new ticker symbols
- Automatic format validation
- Prevents duplicates

#### Remove Tickers
- Select and remove unwanted tickers
- Updates the ticker list immediately

#### Import/Export
- **Export**: Download current ticker list as `tickers.txt`
- **Import**: Upload a text file with ticker symbols
- Validates all imported tickers

#### Reset
- Restore to default ticker list
- Useful for starting fresh

### 4. Ticker Validation
All ticker symbols are validated for:
- Length (1-20 characters)
- Valid characters (letters, numbers, dots, hyphens, underscores)
- Non-empty values

## File Structure

### tickers.txt
```
RELIANCE.NS
TCS.NS
HDFCBANK.NS
SBIN.NS
INFY.NS
...
```

### Default Tickers
The application includes these default Indian stock tickers:
- RELIANCE.NS, TCS.NS, HDFCBANK.NS, SBIN.NS, INFY.NS
- ICICIBANK.NS, HINDUNILVR.NS, ITC.NS, KOTAKBANK.NS, LT.NS
- BAJFINANCE.NS, ASIANPAINT.NS, MARUTI.NS, HCLTECH.NS, AXISBANK.NS

## Usage Examples

### Adding Custom Tickers
1. Select "Enter Custom Ticker" option
2. Type ticker symbol (e.g., "AAPL", "GOOGL", "TSLA")
3. System validates format automatically

### Managing Ticker Lists
1. Open "Ticker Management" expander
2. Add new tickers using the text input
3. Remove unwanted tickers from dropdown
4. Export your custom list for backup
5. Import ticker lists from other sources

### Demo Mode
- Uses the first ticker from available list
- Automatically adapts to your ticker configuration
- No hardcoded ticker dependencies

## Technical Implementation

### Key Functions
- `get_available_tickers()`: Loads tickers from file or defaults
- `validate_ticker_format()`: Validates ticker symbol format
- `save_tickers_to_file()`: Saves ticker list to file
- `load_tickers_from_file()`: Loads ticker list from file

### Error Handling
- File not found: Falls back to defaults
- Invalid formats: Shows specific error messages
- Import errors: Validates and reports issues

## Benefits

1. **Flexibility**: No hardcoded ticker symbols
2. **Extensibility**: Easy to add new markets/exchanges
3. **User Control**: Manage your own ticker lists
4. **Validation**: Prevents invalid ticker formats
5. **Persistence**: Ticker lists saved between sessions
6. **Portability**: Export/import ticker configurations

## Migration from Hardcoded Tickers

The application previously used hardcoded ticker lists:
```python
# OLD: Hardcoded
ticker = st.sidebar.selectbox("Ticker Symbol", ["RELIANCE.NS","TCS.NS","HDFCBANK.NS"])

# NEW: Dynamic
available_tickers = get_available_tickers()
ticker = st.sidebar.selectbox("Ticker Symbol", available_tickers)
```

All hardcoded references have been replaced with dynamic loading functions.