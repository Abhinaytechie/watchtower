import numpy as np
import pandas as pd

def generate_cpg_timeseries(
    n_products: int = 20,
    start: str = "2024-01-01",
    periods: int = 365,
    freq: str = "D",  # "D", "W", "M"
    anomaly_prob: float = 0.02,  # probability per product-date to induce an anomaly in SALES
    seed: int | None = 42,
    promo_prob: float = 0.08,    # probability of a promo on a given date
    price_volatility: float = 0.02,  # daily/periodic price random-walk volatility
    elasticities: tuple[float, float] = (-1.4, -0.3),  # random elasticity range (negative)
    base_price_range: tuple[float, float] = (40.0, 400.0),  # per-unit base price range
    base_demand_range: tuple[float, float] = (30.0, 800.0), # average units per period
) -> pd.DataFrame:
    """
    Generate synthetic time series for a CPG company.

    Returns a DataFrame with columns:
      ['date','product_id','price','promo','sales','is_anomaly']

    Parameters
    ----------
    n_products : number of distinct products
    start      : start date (inclusive)
    periods    : number of time periods to generate
    freq       : pandas frequency alias ("D", "W", "M")
    anomaly_prob : probability per (product, date) to induce a SALES anomaly
    seed       : RNG seed for reproducibility
    promo_prob : per-period promo probability
    price_volatility : random-walk volatility for price (per period)
    elasticities : (min,max) for random price elasticity per product (negative values)
    base_price_range : (min,max) starting price per product
    base_demand_range: (min,max) base demand per product (units per period)
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dates = pd.date_range(start=start, periods=periods, freq=freq)
    # Seasonal handles
    if freq == "D":
        # Weekly and annual seasonality
        dow = pd.Index(dates.dayofweek)  # 0=Mon .. 6=Sun
        day_of_year = pd.Index(dates.dayofyear)
        weekly_season = 0.1 * np.sin(2 * np.pi * (dow / 7.0 - 0.25))  # mild weekly bumps
        # Annual cycle only makes sense for daily
        annual_season = 0.15 * np.sin(2 * np.pi * day_of_year / 365.25)
        base_seasonal = weekly_season.values + annual_season.values
    elif freq == "W":
        # Keep just a mild annual-ish cycle across weeks
        idx = np.arange(periods)
        base_seasonal = 0.12 * np.sin(2 * np.pi * idx / 52.0)
    elif freq == "M":
        # Month-of-year seasonality
        moy = pd.Index(dates.month)
        base_seasonal = 0.18 * np.sin(2 * np.pi * (moy / 12.0 - 0.2))
    else:
        # Generic mild seasonality
        idx = np.arange(periods)
        base_seasonal = 0.10 * np.sin(2 * np.pi * idx / max(6, periods))

    records = []

    # Prepick product-level parameters
    base_prices = rng.uniform(base_price_range[0], base_price_range[1], size=n_products)
    base_demands = rng.uniform(base_demand_range[0], base_demand_range[1], size=n_products)
    # Negative elasticities (greater magnitude => more sensitive to price)
    prod_elasticities = rng.uniform(elasticities[0], elasticities[1], size=n_products)

    # Promo discount distribution and lift
    # When promo happens, price drops and there's an additional demand lift besides price effect
    promo_discount_mu, promo_discount_sigma = 0.20, 0.07  # ~20% avg price cut
    promo_extra_lift_mu, promo_extra_lift_sigma = 0.15, 0.08  # additive extra demand lift

    for p in range(n_products):
        product_id = f"P{p+1:03d}"

        # Start each product around its base price
        prices = np.empty(periods, dtype=float)
        prices[0] = base_prices[p] * rng.uniform(0.95, 1.05)

        # Random-walk price path (geometric)
        for t in range(1, periods):
            drift = -0.25 * price_volatility**2  # tiny drift to avoid explosive walk
            shock = price_volatility * rng.standard_normal()
            prices[t] = prices[t-1] * np.exp(drift + shock)
        # Bound prices to a reasonable band around product base
        prices = np.clip(prices, 0.5 * base_prices[p], 2.0 * base_prices[p])

        # Promo flags
        promo = rng.uniform(0, 1, size=periods) < promo_prob
        # Apply promo discounts (multiplicative) when promo is True
        discounts = np.where(
            promo,
            np.clip(rng.normal(promo_discount_mu, promo_discount_sigma, size=periods), 0.05, 0.5),
            0.0,
        )
        promo_prices = prices * (1.0 - discounts)

        # Demand model
        #   - base demand per product
        #   - multiplicative seasonality (1 + seasonal)
        #   - price elasticity effect: (price / base_price)^elasticity
        #   - promo extra lift (additive on the multiplicative scale)
        #   - lognormal noise for count-like variability
        elasticity = prod_elasticities[p]

        seasonal_mult = 1.0 + base_seasonal  # can go below/above 1
        seasonal_mult = np.clip(seasonal_mult, 0.2, None)

        price_effect = (promo_prices / base_prices[p]) ** elasticity
        extra_lift = np.where(
            promo,
            np.clip(rng.normal(promo_extra_lift_mu, promo_extra_lift_sigma, size=periods), 0.0, 0.5),
            0.0,
        )

        expected_sales = base_demands[p] * seasonal_mult * price_effect * (1.0 + extra_lift)

        # Lognormal multiplicative noise (centered at 1)
        noise = np.exp(rng.normal(loc=0.0, scale=0.25, size=periods))
        sales = expected_sales * noise

        # Round to realistic integers
        sales = np.round(np.clip(sales, 0.0, None)).astype(int)

        # Anomaly injection on SALES
        is_anomaly = np.zeros(periods, dtype=bool)
        anomaly_flags = rng.uniform(0, 1, size=periods) < anomaly_prob

        # Types of anomalies (probabilities sum to 1)
        #   spike: sudden surge
        #   drop: sudden drop but non-zero
        #   zero_out: stockout or data miss
        #   burst_zero: small run of zeros (stockout window)
        anomaly_types = ["spike", "drop", "zero_out", "burst_zero"]
        anomaly_weights = np.array([0.45, 0.35, 0.15, 0.05])

        t = 0
        while t < periods:
            if anomaly_flags[t]:
                atype = rng.choice(anomaly_types, p=anomaly_weights)
                is_anomaly[t] = True

                if atype == "spike":
                    factor = rng.uniform(2.0, 5.0)
                    sales[t] = int(np.round(sales[t] * factor))

                elif atype == "drop":
                    factor = rng.uniform(0.1, 0.5)
                    sales[t] = int(np.round(sales[t] * factor))

                elif atype == "zero_out":
                    sales[t] = 0

                elif atype == "burst_zero":
                    # 2–5 consecutive zeros if room remains
                    length = int(rng.integers(2, 6))
                    end = min(periods, t + length)
                    sales[t:end] = 0
                    is_anomaly[t:end] = True
                    t = end - 1  # jump to end-1; loop will increment to end
            t += 1

        # Assemble rows
        for i, dt in enumerate(dates):
            records.append({
                "date": dt,
                "product_id": product_id,
                "price": round(float(promo_prices[i]), 2),
                "promo": bool(promo[i]),
                "sales": int(sales[i]),
                "is_anomaly": bool(is_anomaly[i]),
            })

    df = pd.DataFrame.from_records(records)
    df.sort_values(["product_id", "date"], inplace=True, ignore_index=True)
    return df


# -------------------------
# Example usage:
if __name__ == "__main__":
    df = generate_cpg_timeseries(
        n_products=5,
        start="2025-01-05", #start 6 months ago
        periods=180,
        freq="D",
        anomaly_prob=0.1,
        seed=123
    )
    df.to_csv("data-gen/synthetic_data.csv", index=False)
    # For quick pivoted view:
    sales_pivot = df.pivot(index="date", columns="product_id", values="sales")
    price_pivot  = df.pivot(index="date", columns="product_id", values="price")
    print(sales_pivot.head(10))
    print(price_pivot.head(10))
