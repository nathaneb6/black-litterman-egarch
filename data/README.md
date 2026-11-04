# Data

The raw price files are **not versioned** because they are Investing.com exports, and Investing.com does not allow redistribution.

To reproduce the results, download the **daily historical data from 2007-01-01 to 2026-03-18** for each of the 10 stocks from [Investing.com](https://www.investing.com). Save them in this folder as `<Asset name>.txt`, using exactly these names:

```
Air Liquide.txt   Airbus.txt   AXA.txt   BNP Paribas.txt   Capgemini.txt
Danone.txt   LVMH.txt   Sanofi.txt   Schneider Electric.txt   TotalEnergies.txt
```

Expected format (comma-separated, any date order):

```
Date,Dernier,Ouv., Plus Haut,Plus Bas,Vol.,Variation %
2026-03-18,173.62,171.98,174.10,171.38,279190.00,1.94
```

Only `Date` and the closing price (`Dernier` or `Close`) are used.

Without data, run `python scripts/run_backtest.py --synthetic` instead.
