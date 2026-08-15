"""Data validation utilities for ML pipeline."""

def validate_dataset(df):
    """Validate dataset before processing."""
    issues = []

    null_counts = df.isnull().sum()
    if null_counts.any():
        issues.append(
            f"Null values found: {null_counts[null_counts > 0].to_dict()}"
        )

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Duplicate rows found: {duplicates}")

    return issues


def clean_dataset(df):
    """Basic dataset cleaning operations."""
    df_clean = df.drop_duplicates()

    numeric_columns = df_clean.select_dtypes(include=["number"]).columns
    df_clean[numeric_columns] = df_clean[numeric_columns].fillna(
        df_clean[numeric_columns].median()
    )

    return df_clean