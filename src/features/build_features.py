
import pandas as pd
import numpy as np


GROUP_COLS = ["store_nbr", "family"]


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int16")

    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_month_start"] = df["date"].dt.is_month_start.astype("int8")
    df["is_month_end"] = df["date"].dt.is_month_end.astype("int8")
    df["is_payday"] = (
        (df["day"] == 15) | df["is_month_end"].eq(1)
    ).astype("int8")

    # Cyclical calendar features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["weekday_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["weekday_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def sales_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)

    grouped = df.groupby(GROUP_COLS, sort=False)["sales"]

    # Historical sales lags
    for lag in [1, 7, 14, 21, 28, 35, 56]:
        df[f"sales_lag_{lag}"] = grouped.shift(lag)

    # Previous-day sales, excluding current-day sales
    previous_sales = df.groupby(GROUP_COLS, sort=False)["sales"].shift(1)

    # Rolling demand statistics
    for window in [7, 14, 28]:
        rolling = (
            previous_sales
            .groupby([df[col] for col in GROUP_COLS], sort=False)
            .rolling(window=window, min_periods=window)
        )

        df[f"sales_mean_{window}"] = (
            rolling.mean()
            .reset_index(level=GROUP_COLS, drop=True)
            .reindex(df.index)
        )

        df[f"sales_std_{window}"] = (
            rolling.std()
            .reset_index(level=GROUP_COLS, drop=True)
            .reindex(df.index)
        )

    # Zero-sales behaviour over the previous 28 observations
    previous_zero = (previous_sales == 0).astype(float)

    df["zero_sales_pct_28"] = (
        previous_zero
        .groupby([df[col] for col in GROUP_COLS], sort=False)
        .rolling(28, min_periods=28)
        .mean()
        .reset_index(level=GROUP_COLS, drop=True)
        .reindex(df.index)
    )

    # Difference between recent and longer-term demand
    df["sales_trend_7_28"] = (
        df["sales_mean_7"] - df["sales_mean_28"]
    )

    return df


def promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["has_promotion"] = (df["onpromotion"] > 0).astype("int8")

    df = df.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)

    grouped = df.groupby(GROUP_COLS, sort=False)["has_promotion"]
    previous_promotion = grouped.shift(1)

    for window in [7, 28]:
        df[f"promotion_days_{window}"] = (
            previous_promotion
            .groupby(
                [df[col] for col in GROUP_COLS],
                sort=False
            )
            .rolling(window, min_periods=window)
            .sum()
            .reset_index(level=GROUP_COLS, drop=True)
            .reindex(df.index)
        )

    return df

def create_forecast_lag_features(
    history: pd.DataFrame,
    forecast_rows: pd.DataFrame
) -> pd.DataFrame:

    group_cols = ["store_nbr", "family"]

    lag_values = [1, 7, 14, 21, 28, 35, 56]
    rolling_windows = [7, 14, 28]

    history = history.sort_values(
        group_cols + ["date"]
    )

    # Keep enough history for the longest lag / rolling window
    history_tail = (
        history
        .groupby(group_cols, sort=False)
        .tail(56)
    )

    sales_by_group = {
        key: group["sales"].to_numpy()
        for key, group in history_tail.groupby(
            group_cols,
            sort=False
        )
    }

    feature_rows = []

    for row in forecast_rows.itertuples(index=False):

        key = (row.store_nbr, row.family)

        past_sales = sales_by_group.get(
            key,
            np.array([], dtype="float32")
        )

        features = row._asdict()

        # -------------------------
        # Sales lags
        # -------------------------
        for lag in lag_values:

            if len(past_sales) >= lag:
                features[f"sales_lag_{lag}"] = past_sales[-lag]
            else:
                features[f"sales_lag_{lag}"] = np.nan

        # -------------------------
        # Rolling features
        # -------------------------
        for window in rolling_windows:

            if len(past_sales) >= window:

                recent = past_sales[-window:]

                features[f"sales_mean_{window}"] = recent.mean()
                features[f"sales_std_{window}"] = recent.std()

                if window == 28:
                    features["zero_sales_pct_28"] = (
                        (recent == 0).mean()
                    )

            else:

                features[f"sales_mean_{window}"] = np.nan
                features[f"sales_std_{window}"] = np.nan

                if window == 28:
                    features["zero_sales_pct_28"] = np.nan

        # -------------------------
        # Demand trend
        # -------------------------
        if (
            len(past_sales) >= 28
        ):
            features["sales_trend_7_28"] = (
                past_sales[-7:].mean()
                - past_sales[-28:].mean()
            )
        else:
            features["sales_trend_7_28"] = np.nan

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)