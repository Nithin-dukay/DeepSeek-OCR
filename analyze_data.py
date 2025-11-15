import pandas as pd
import numpy as np
import json
from collections import Counter

# Read the complete CSV file
print("Loading employment data...")
df = pd.read_csv('/vercel/sandbox/uploads/employment-data.csv')

# Basic dataset information
print("\n" + "="*80)
print("PHASE 1: DATASET STRUCTURE DISCOVERY")
print("="*80)

print(f"\nTotal Rows: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")
print(f"\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nData Types:")
print(df.dtypes)

print(f"\nMemory Usage:")
print(df.memory_usage(deep=True))

# Check for missing values
print(f"\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

# Unique values in key columns
print("\n" + "="*80)
print("PHASE 2: UNIQUE VALUES ANALYSIS")
print("="*80)

key_columns = ['Series_reference', 'STATUS', 'UNITS', 'Magnitude', 'Subject', 
               'Group', 'Series_title_1', 'Series_title_2', 'Series_title_3']

for col in key_columns:
    if col in df.columns:
        unique_count = df[col].nunique()
        print(f"\n{col}: {unique_count} unique values")
        if unique_count <= 20:
            print(f"  Values: {sorted(df[col].dropna().unique().tolist())}")
        else:
            print(f"  Sample values: {df[col].dropna().unique()[:10].tolist()}")

# Analyze Period column
print("\n" + "="*80)
print("PHASE 3: TIME PERIOD ANALYSIS")
print("="*80)

if 'Period' in df.columns:
    df['Period'] = df['Period'].astype(str)
    periods = sorted(df['Period'].unique())
    print(f"\nTotal Periods: {len(periods)}")
    print(f"First Period: {periods[0]}")
    print(f"Last Period: {periods[-1]}")
    print(f"\nSample Periods: {periods[:10]}")

# Analyze Data_value column
print("\n" + "="*80)
print("PHASE 4: DATA VALUE ANALYSIS")
print("="*80)

if 'Data_value' in df.columns:
    # Convert to numeric, handling empty strings
    df['Data_value_numeric'] = pd.to_numeric(df['Data_value'], errors='coerce')
    
    non_null_values = df['Data_value_numeric'].dropna()
    print(f"\nTotal Data Values: {len(df['Data_value'])}")
    print(f"Non-null Numeric Values: {len(non_null_values)}")
    print(f"Null/Suppressed Values: {df['Data_value_numeric'].isnull().sum()}")
    
    if len(non_null_values) > 0:
        print(f"\nStatistics for Numeric Values:")
        print(f"  Min: {non_null_values.min():,.2f}")
        print(f"  Max: {non_null_values.max():,.2f}")
        print(f"  Mean: {non_null_values.mean():,.2f}")
        print(f"  Median: {non_null_values.median():,.2f}")
        print(f"  Std Dev: {non_null_values.std():,.2f}")

# Analyze Suppressed column
print("\n" + "="*80)
print("PHASE 5: SUPPRESSION ANALYSIS")
print("="*80)

if 'Suppressed' in df.columns:
    suppressed_counts = df['Suppressed'].value_counts(dropna=False)
    print(f"\nSuppressed Values Distribution:")
    print(suppressed_counts)
    
    # Check correlation with empty Data_value
    if 'Data_value' in df.columns:
        empty_data = df['Data_value'] == ''
        suppressed_y = df['Suppressed'] == 'Y'
        print(f"\nRows with empty Data_value: {empty_data.sum()}")
        print(f"Rows with Suppressed='Y': {suppressed_y.sum()}")
        print(f"Rows with both: {(empty_data & suppressed_y).sum()}")

# Save detailed analysis to JSON
print("\n" + "="*80)
print("PHASE 6: SAVING DETAILED INSIGHTS")
print("="*80)

insights = {
    "dataset_overview": {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    },
    "unique_values": {},
    "time_period": {
        "total_periods": int(len(periods)) if 'Period' in df.columns else 0,
        "first_period": periods[0] if 'Period' in df.columns and len(periods) > 0 else None,
        "last_period": periods[-1] if 'Period' in df.columns and len(periods) > 0 else None,
        "all_periods": periods if 'Period' in df.columns else []
    },
    "data_values": {
        "total": int(len(df['Data_value'])) if 'Data_value' in df.columns else 0,
        "non_null": int(len(non_null_values)) if 'Data_value' in df.columns else 0,
        "null_count": int(df['Data_value_numeric'].isnull().sum()) if 'Data_value' in df.columns else 0
    }
}

# Add unique values for each column
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    insights["unique_values"][col] = {
        "count": int(len(unique_vals)),
        "values": unique_vals.tolist()[:100] if len(unique_vals) <= 100 else unique_vals.tolist()[:50]
    }

with open('/vercel/sandbox/analysis_insights.json', 'w') as f:
    json.dump(insights, f, indent=2, default=str)

print("\nInsights saved to: analysis_insights.json")
print("\nPhase 1 Analysis Complete!")

# PHASE 2: DEEP ANALYSIS
print("\n" + "="*80)
print("PHASE 2: DEEP ANALYSIS - PATTERNS AND GROUPINGS")
print("="*80)

# Analyze by Group
print("\n--- Analysis by Group ---")
group_analysis = {}
for group in df['Group'].unique():
    group_df = df[df['Group'] == group]
    group_analysis[group] = {
        "total_records": int(len(group_df)),
        "unique_series": int(group_df['Series_reference'].nunique()),
        "unique_categories": int(group_df['Series_title_2'].nunique()),
        "date_range": f"{group_df['Period'].min()} to {group_df['Period'].max()}"
    }
    print(f"\n{group}:")
    print(f"  Records: {group_analysis[group]['total_records']:,}")
    print(f"  Unique Series: {group_analysis[group]['unique_series']}")
    print(f"  Categories: {group_analysis[group]['unique_categories']}")
    print(f"  Date Range: {group_analysis[group]['date_range']}")

# Analyze Industries (for Industry by employment variable)
print("\n--- Industry Analysis ---")
industry_df = df[df['Group'] == 'Industry by employment variable']
if len(industry_df) > 0:
    industries = industry_df['Series_title_2'].unique()
    print(f"\nTotal Industries: {len(industries)}")
    print("Industries:")
    for ind in sorted(industries):
        count = len(industry_df[industry_df['Series_title_2'] == ind])
        print(f"  - {ind}: {count:,} records")

# Analyze Age Groups
print("\n--- Age Group Analysis ---")
age_df = df[df['Group'] == 'Age by employment variable']
if len(age_df) > 0:
    age_groups = age_df['Series_title_2'].unique()
    print(f"\nTotal Age Groups: {len(age_groups)}")
    print("Age Groups:")
    for age in sorted(age_groups):
        count = len(age_df[age_df['Series_title_2'] == age])
        print(f"  - {age}: {count:,} records")

# Analyze Regions
print("\n--- Region Analysis ---")
region_df = df[df['Group'] == 'Region by employment variable']
if len(region_df) > 0:
    regions = region_df['Series_title_2'].unique()
    print(f"\nTotal Regions: {len(regions)}")
    print("Regions (sample):")
    for region in sorted(regions)[:20]:
        count = len(region_df[region_df['Series_title_2'] == region])
        print(f"  - {region}: {count:,} records")

# Analyze measurement types (Actual, Seasonally adjusted, Trend)
print("\n--- Measurement Type Analysis ---")
measurement_counts = df['Series_title_3'].value_counts()
print("\nMeasurement Types:")
print(measurement_counts)

# Analyze UNITS and what they measure
print("\n--- Units Analysis ---")
units_analysis = df.groupby(['UNITS', 'Series_title_1']).size().reset_index(name='count')
print("\nUnits by Measurement Type:")
print(units_analysis)

# Analyze STATUS codes
print("\n--- STATUS Code Analysis ---")
status_counts = df['STATUS'].value_counts()
print("\nSTATUS Codes:")
for status, count in status_counts.items():
    print(f"  {status}: {count:,} records")

print("\nPhase 2 Analysis Complete!")

# PHASE 3: STATISTICAL ANALYSIS AND TRENDS
print("\n" + "="*80)
print("PHASE 3: STATISTICAL ANALYSIS AND TRENDS")
print("="*80)

# Analyze filled jobs trends over time
print("\n--- Filled Jobs Trends by Industry (Seasonally Adjusted) ---")
filled_jobs_df = df[(df['Series_title_1'] == 'Filled jobs') & 
                     (df['Series_title_3'] == 'Seasonally adjusted') &
                     (df['Group'] == 'Industry by employment variable')]

if len(filled_jobs_df) > 0:
    # Group by industry and calculate statistics
    industry_stats = {}
    for industry in filled_jobs_df['Series_title_2'].unique():
        ind_data = filled_jobs_df[filled_jobs_df['Series_title_2'] == industry]['Data_value'].dropna()
        if len(ind_data) > 0:
            industry_stats[industry] = {
                "count": int(len(ind_data)),
                "mean": float(ind_data.mean()),
                "min": float(ind_data.min()),
                "max": float(ind_data.max()),
                "std": float(ind_data.std()),
                "growth": float(((ind_data.iloc[-1] - ind_data.iloc[0]) / ind_data.iloc[0] * 100)) if len(ind_data) > 1 else 0
            }
    
    print("\nIndustry Employment Statistics (Seasonally Adjusted):")
    for industry, stats in sorted(industry_stats.items(), key=lambda x: x[1]['mean'], reverse=True):
        print(f"\n{industry}:")
        print(f"  Average Jobs: {stats['mean']:,.0f}")
        print(f"  Range: {stats['min']:,.0f} - {stats['max']:,.0f}")
        print(f"  Growth (2011-2025): {stats['growth']:+.1f}%")

# Analyze total earnings trends
print("\n--- Total Earnings Trends by Industry (Actual) ---")
earnings_df = df[(df['Series_title_1'] == 'Total earnings') & 
                  (df['Series_title_3'] == 'Actual') &
                  (df['Group'] == 'Industry by employment variable')]

if len(earnings_df) > 0:
    earnings_stats = {}
    for industry in earnings_df['Series_title_2'].unique():
        ind_data = earnings_df[earnings_df['Series_title_2'] == industry]['Data_value'].dropna()
        if len(ind_data) > 0:
            earnings_stats[industry] = {
                "count": int(len(ind_data)),
                "mean": float(ind_data.mean()),
                "min": float(ind_data.min()),
                "max": float(ind_data.max()),
                "latest": float(ind_data.iloc[-1]) if len(ind_data) > 0 else 0,
                "growth": float(((ind_data.iloc[-1] - ind_data.iloc[0]) / ind_data.iloc[0] * 100)) if len(ind_data) > 1 else 0
            }
    
    print("\nIndustry Earnings Statistics (in millions):")
    for industry, stats in sorted(earnings_stats.items(), key=lambda x: x[1]['latest'], reverse=True):
        print(f"\n{industry}:")
        print(f"  Latest Earnings: ${stats['latest']:,.2f}M")
        print(f"  Average: ${stats['mean']:,.2f}M")
        print(f"  Growth (2011-2025): {stats['growth']:+.1f}%")

# Analyze by Sex
print("\n--- Employment by Sex (Seasonally Adjusted) ---")
sex_df = df[(df['Group'] == 'Sex by employment variable') & 
            (df['Series_title_1'] == 'Filled jobs') &
            (df['Series_title_3'] == 'Actual')]

if len(sex_df) > 0:
    for sex in sex_df['Series_title_2'].unique():
        sex_data = sex_df[sex_df['Series_title_2'] == sex]['Data_value'].dropna()
        if len(sex_data) > 0:
            print(f"\n{sex}:")
            print(f"  Average Jobs: {sex_data.mean():,.0f}")
            print(f"  Latest (2025.06): {sex_data.iloc[-1]:,.0f}")
            print(f"  Growth: {((sex_data.iloc[-1] - sex_data.iloc[0]) / sex_data.iloc[0] * 100):+.1f}%")

# Analyze by Age Groups
print("\n--- Employment by Age Group (Actual) ---")
age_jobs_df = df[(df['Group'] == 'Age by employment variable') & 
                  (df['Series_title_1'] == 'Filled jobs') &
                  (df['Series_title_3'] == 'Actual')]

if len(age_jobs_df) > 0:
    age_stats = {}
    for age in age_jobs_df['Series_title_2'].unique():
        age_data = age_jobs_df[age_jobs_df['Series_title_2'] == age]['Data_value'].dropna()
        if len(age_data) > 0:
            age_stats[age] = {
                "mean": float(age_data.mean()),
                "latest": float(age_data.iloc[-1]),
                "growth": float(((age_data.iloc[-1] - age_data.iloc[0]) / age_data.iloc[0] * 100))
            }
    
    print("\nAge Group Employment Statistics:")
    for age, stats in sorted(age_stats.items()):
        print(f"\n{age}:")
        print(f"  Average Jobs: {stats['mean']:,.0f}")
        print(f"  Latest (2025.06): {stats['latest']:,.0f}")
        print(f"  Growth: {stats['growth']:+.1f}%")

print("\nPhase 3 Analysis Complete!")
