import pandas as pd

def convert_date_columns(data: dict) -> dict:
    data_clean = {}

    datasets_with_dates = [
        "train",
        "test",
        "oil",
        "holidays_events",
        "transactions",
    ]

    for name, df in data.items():
        df_clean = df.copy()

        if name in datasets_with_dates:
            df_clean["date"] = pd.to_datetime(df_clean["date"])

        data_clean[name] = df_clean

    return data_clean



def filling_oil_missing_values(oil: pd.DataFrame) -> pd.DataFrame:
    oil_filled = oil.copy()

    # Fill missing values using the previous available price
    oil_filled["dcoilwtico"] = oil_filled["dcoilwtico"].ffill()

    # Handle any remaining missing value at the beginning
    oil_filled["dcoilwtico"] = oil_filled["dcoilwtico"].bfill(limit=1)

    return oil_filled

def cap_promotion_outliers(train: pd.DataFrame,cap: int = 169) -> pd.DataFrame:
    train_clean = train.copy()

    train_clean["onpromotion"] = train_clean["onpromotion"].clip(upper=cap)
    
    return train_clean

def cleaning_holidays_events(holidays_events: pd.DataFrame) -> pd.DataFrame:
    holidays_events_clean = holidays_events.copy()
    holiday_clean_df = holidays_events_clean.groupby("date", as_index=False).agg(
                        event_count= ("type", "count"),
                        has_holiday= ("type", lambda x: (x == "Holiday").any()),
                        has_event= ("type", lambda x: (x == "Event").any()),
                        has_additional= ("type", lambda x: (x == "Additional").any()),
                        has_transfer= ("type", lambda x: (x == "Transfer").any()),
                        has_bridge= ("type", lambda x: (x == "Bridge").any()),
                        has_workday= ("type", lambda x: (x == "Work Day").any()),
                        is_national= ("locale", lambda x: (x == "National").any()),
                        is_regional= ("locale", lambda x: (x == "Regional").any()),
                        is_local= ("locale", lambda x: (x == "Local").any()),   
                    )
    return holiday_clean_df


def cleaned_data(data: dict) -> dict:
    data_clean = convert_date_columns(data)
    data_clean["oil"] = filling_oil_missing_values(data_clean["oil"])
    data_clean["train"] = cap_promotion_outliers(data_clean["train"])
    data_clean["holidays_events"] = cleaning_holidays_events(data_clean["holidays_events"])
    return data_clean

def master_data(data: dict, dataset: str = "train") -> pd.DataFrame:

    merged_df = (
        data[dataset]
        .merge(data["oil"], on="date", how="left")
        .merge(data["holidays_events"], on="date", how="left")
        .merge(data["stores"], on="store_nbr", how="left")
    )

    # Transactions exist only for historical training dates
    if dataset == "train":
        merged_df = merged_df.merge(
            data["transactions"],
            on=["date", "store_nbr"],
            how="left"
        )
    else:
        merged_df["transactions"] = float("nan")

    return merged_df


def clean_master_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    holiday_num_col = ['event_count']
    holiday_bool_cols = [
        'has_holiday', 'has_event', 'has_additional',
        'has_transfer', 'has_bridge', 'has_workday'
    ]
    locale_bool_cols = ['is_local', 'is_regional', 'is_national']

    df[holiday_num_col] = df[holiday_num_col].fillna(0)
    df[holiday_bool_cols] = df[holiday_bool_cols].fillna(False)
    df[locale_bool_cols] = df[locale_bool_cols].fillna(False)

    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()

    return df
