EXPECTED_COLUMNS = {
    "oil": ["date", "dcoilwtico"],
    "stores": ["store_nbr", "city", "state", "type", "cluster"],
    "train": ["id","date", "store_nbr", "family", "sales", "onpromotion"],
    "holidays_events": ['date', 'type', 'locale', 'locale_name', 'description', 'transferred'],
    "transactions": ["date", "store_nbr", "transactions"],
    "test": ["id","date", "store_nbr", "family","onpromotion"]
    }

def validate_schema(data):
    
    validation_results = {}
    
    for name, df in data.items():
        expected_cols = EXPECTED_COLUMNS.get(name)
        if expected_cols is None:
            validation_results[name] = {"status": "error", "message": f"No expected columns defined for {name}"}
            continue
        
        actual_cols = df.columns.tolist()
        
        if set(expected_cols) == set(actual_cols):
            validation_results[name] = {"status": "success", "message": "Schema validation passed"}
        else:
            missing_cols = set(expected_cols) - set(actual_cols)
            extra_cols = set(actual_cols) - set(expected_cols)
            validation_results[name] = {
                "status": "error",
                "message": f"Schema validation failed. Missing columns: {missing_cols}, Extra columns: {extra_cols}"
            }
    return validation_results
    
def validate_missing_values(data):
    missing_values_results = {}
    
    for name, df in data.items():
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            missing_values_results[name] = {"status": "success", "message": "No missing values"}
        else:
            missing_values_results[name] = {"status": "error", "message": f"Missing values found: {missing_count}"}
    return missing_values_results

def validate_duplicates(data):
    duplicates_results = {}
    
    for name, df in data.items():
        duplicate_count = df.duplicated().sum()
        if duplicate_count == 0:
            duplicates_results[name] = {"status": "success", "message": "No duplicate rows"}
        else:
            duplicates_results[name] = {"status": "error", "message": f"Duplicate rows found: {duplicate_count}"}
    return duplicates_results


def validate_business_keys(data):
    business_keys_results = {}

    for name, df in data.items():

        if name in ["train", "test"]:
            key_columns = ["date", "store_nbr", "family"]

        elif name == "transactions":
            key_columns = ["date", "store_nbr"]

        elif name == "oil":
            key_columns = ["date"]

        elif name == "stores":
            key_columns = ["store_nbr"]

        else:
            # No business key defined for this dataset
            continue

        duplicate_count = df.duplicated(subset=key_columns).sum()

        if duplicate_count == 0:
            business_keys_results[name] = {
                "status": "success",
                "message": "No duplicate business keys"
            }
        else:
            business_keys_results[name] = {
                "status": "error",
                "message": (
                    f"Duplicate business keys found: {duplicate_count}"
                )
            }

    return business_keys_results



def validate_store_relationship(data):
    store_relationship_results = {}

    if "stores" not in data:
        return {
            "stores": {
                "status": "error",
                "message": "Stores dataset is missing"
            }
        }

    stores_master = set(data["stores"]["store_nbr"].unique())

    for name in ["train", "transactions"]:
        if name not in data:
            continue

        referenced_stores = set(data[name]["store_nbr"].unique())
        missing_stores = referenced_stores - stores_master

        if not missing_stores:
            store_relationship_results[name] = {
                "status": "success",
                "message": "All referenced stores exist in stores"
            }
        else:
            store_relationship_results[name] = {
                "status": "error",
                "message": f"Missing store_nbr values: {missing_stores}"
            }

    return store_relationship_results


def validate_date_coverage(data):
    date_coverage_results = {}

    if "train" in data and "test" in data:
        train_dates = set(data["train"]["date"].unique())
        test_dates = set(data["test"]["date"].unique())

        if train_dates and test_dates and max(train_dates) < min(test_dates):
            date_coverage_results["train_test"] = {
                "status": "success",
                "message": "Train and test periods are chronologically separated"
            }
        else:
            date_coverage_results["train_test"] = {
                "status": "error",
                "message": "Train and test periods overlap or have invalid date ranges"
            }

    if "oil" in data and "train" in data and "test" in data:
        required_dates = (
            set(data["train"]["date"].unique())
            | set(data["test"]["date"].unique())
        )
        oil_dates = set(data["oil"]["date"].unique())

        missing_oil_dates = required_dates - oil_dates

        if not missing_oil_dates:
            date_coverage_results["oil"] = {
                "status": "success",
                "message": "Oil dataset contains every required calendar date"
            }
        else:
            date_coverage_results["oil"] = {
                "status": "error",
                "message": f"Missing oil dates: {missing_oil_dates}"
            }

    return date_coverage_results

