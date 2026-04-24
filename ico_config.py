# ICO dataset configuration — standalone, no heavy dependencies.
# Copy this file into any project that needs to work with ICO label strategies.

# Columns in the raw ICO CSV that are metadata (not features for the LLM/model).
ICO_METADATA_COLUMNS = ['riskLevel', 'name', 'token_symbol', 'dc_Category', 'dc_EntryDate', 'dc_Summary']
ICO_METADATA_PREFIX = 'meta_'

# Label strategy definitions for the ICO dataset.
#   positive : riskLevel values treated as fraud (label=1)
#   drop     : riskLevel values excluded from the dataset entirely
#
# 'all'       — default; all non-zero risk levels are fraud
# 'high_only' — only riskLevel 2 & 3 are fraud; riskLevel 1 rows are dropped
# 'low_only'  — only riskLevel 1 is fraud;       riskLevel 2 & 3 rows are dropped
ICO_LABEL_STRATEGIES = {
    'all':       {'positive': [1, 2, 3], 'drop': []},
    'high_only': {'positive': [2, 3],    'drop': [1]},
    'low_only':  {'positive': [1],       'drop': [2, 3]},
}


def is_metadata_column(column_name):
    return column_name.startswith(ICO_METADATA_PREFIX)


def apply_ico_label_strategy(df, strategy_name):
    """Filter rows and assign binary label according to the named ICO label strategy."""
    if strategy_name not in ICO_LABEL_STRATEGIES:
        raise ValueError(f"Unknown ico_label_strategy '{strategy_name}'. "
                         f"Choose from: {list(ICO_LABEL_STRATEGIES.keys())}")
    strategy = ICO_LABEL_STRATEGIES[strategy_name]
    if strategy['drop']:
        df = df[~df['riskLevel'].isin(strategy['drop'])].copy()
    else:
        df = df.copy()
    df['label'] = df['riskLevel'].isin(strategy['positive']).astype(int)
    return df
