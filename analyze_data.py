#!/usr/bin/env python3
"""
Employment Data Analysis Script - Phase 3: Trend Analysis & Statistical Insights
This script performs comprehensive analysis of the employment-data.csv file
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import Counter

# Load the complete dataset
print("=" * 80)
print("PHASE 3: TREND ANALYSIS & STATISTICAL INSIGHTS")
print("=" * 80)

df = pd.read_csv('/vercel/sandbox/uploads/employment-data.csv')
df['Data_value_numeric'] = pd.to_numeric(df['Data_value'], errors='coerce')

print(f"\n✓ Dataset loaded: {len(df):,} rows")

# Analyze by Group
print("\n" + "=" * 80)
print("ANALYSIS BY GROUP")
print("=" * 80)

for group in df['Group'].unique():
    group_df = df[df['Group'] == group]
    print(f"\n{group}:")
    print(f"  Total records: {len(group_df):,}")
    print(f"  Unique series: {group_df['Series_reference'].nunique()}")
    print(f"  Categories (Series_title_2): {group_df['Series_title_2'].nunique()}")
    print(f"  Top 5 categories:")
    top_cats = group_df['Series_title_2'].value_counts().head(5)
    for cat, count in top_cats.items():
        print(f"    • {cat}: {count:,} records")

# Analyze by Series_title_1 (Employment Variable)
print("\n" + "=" * 80)
print("ANALYSIS BY EMPLOYMENT VARIABLE (Series_title_1)")
print("=" * 80)

for var in df['Series_title_1'].unique():
    var_df = df[df['Series_title_1'] == var]
    print(f"\n{var}:")
    print(f"  Total records: {len(var_df):,}")
    print(f"  Non-null values: {var_df['Data_value_numeric'].notna().sum():,}")
    print(f"  Suppressed values: {(var_df['Suppressed'] == 'Y').sum():,}")
    
    if var_df['Data_value_numeric'].notna().sum() > 0:
        print(f"  Value range: {var_df['Data_value_numeric'].min():,.0f} to {var_df['Data_value_numeric'].max():,.0f}")
        print(f"  Average: {var_df['Data_value_numeric'].mean():,.0f}")

# Analyze by Series_title_3 (Adjustment Type)
print("\n" + "=" * 80)
print("ANALYSIS BY ADJUSTMENT TYPE (Series_title_3)")
print("=" * 80)

for adj_type in df['Series_title_3'].unique():
    adj_df = df[df['Series_title_3'] == adj_type]
    print(f"\n{adj_type}:")
    print(f"  Total records: {len(adj_df):,}")
    print(f"  Percentage of total: {(len(adj_df) / len(df) * 100):.2f}%")
    print(f"  STATUS distribution:")
    for status, count in adj_df['STATUS'].value_counts().items():
        print(f"    • {status}: {count:,} ({count/len(adj_df)*100:.1f}%)")

# Time series analysis
print("\n" + "=" * 80)
print("TIME SERIES ANALYSIS")
print("=" * 80)

df['Year'] = df['Period'].apply(lambda x: int(x) if pd.notna(x) else None)
df['Quarter'] = df['Period'].apply(lambda x: int((x % 1) * 100) if pd.notna(x) else None)

print(f"\nYears covered: {sorted(df['Year'].dropna().unique().astype(int).tolist())}")
print(f"Quarters: {sorted(df['Quarter'].dropna().unique().astype(int).tolist())}")

# Records per year
print("\nRecords per year:")
year_counts = df['Year'].value_counts().sort_index()
for year, count in year_counts.items():
    if pd.notna(year):
        print(f"  {int(year)}: {count:,} records")

# Industry analysis (for Industry by employment variable group)
print("\n" + "=" * 80)
print("INDUSTRY ANALYSIS")
print("=" * 80)

industry_df = df[df['Group'] == 'Industry by employment variable']
print(f"\nTotal industry records: {len(industry_df):,}")
print(f"\nIndustries tracked ({industry_df['Series_title_2'].nunique()} unique):")

industries = industry_df['Series_title_2'].unique()
for industry in sorted(industries):
    ind_data = industry_df[industry_df['Series_title_2'] == industry]
    filled_jobs = ind_data[ind_data['Series_title_1'] == 'Filled jobs']
    if len(filled_jobs) > 0:
        latest_value = filled_jobs.sort_values('Period', ascending=False).iloc[0]['Data_value_numeric']
        print(f"  • {industry}: Latest = {latest_value:,.0f}" if pd.notna(latest_value) else f"  • {industry}: Latest = N/A")

# Age group analysis
print("\n" + "=" * 80)
print("AGE GROUP ANALYSIS")
print("=" * 80)

age_df = df[df['Group'] == 'Age by employment variable']
print(f"\nTotal age group records: {len(age_df):,}")
print(f"Age groups tracked: {sorted(age_df['Series_title_2'].unique())}")

# Regional analysis
print("\n" + "=" * 80)
print("REGIONAL ANALYSIS")
print("=" * 80)

region_df = df[df['Group'] == 'Region by employment variable']
print(f"\nTotal regional records: {len(region_df):,}")
print(f"Regions tracked: {region_df['Series_title_2'].nunique()}")
print(f"Sample regions: {sorted(region_df['Series_title_2'].unique())[:10]}")

# Sex analysis
print("\n" + "=" * 80)
print("SEX-BASED ANALYSIS")
print("=" * 80)

sex_df = df[df['Group'] == 'Sex by employment variable']
print(f"\nTotal sex-based records: {len(sex_df):,}")
print(f"Categories: {sorted(sex_df['Series_title_2'].unique())}")

# Trend Analysis - Industry Employment Over Time
print("\n" + "=" * 80)
print("INDUSTRY EMPLOYMENT TRENDS (2011-2025)")
print("=" * 80)

industry_filled = industry_df[
    (industry_df['Series_title_1'] == 'Filled jobs') & 
    (industry_df['Series_title_3'] == 'Actual')
].copy()

industry_trends = {}
for industry in sorted(industry_filled['Series_title_2'].unique()):
    ind_data = industry_filled[industry_filled['Series_title_2'] == industry].sort_values('Period')
    if len(ind_data) > 0:
        first_val = ind_data.iloc[0]['Data_value_numeric']
        last_val = ind_data.iloc[-1]['Data_value_numeric']
        if pd.notna(first_val) and pd.notna(last_val) and first_val > 0:
            pct_change = ((last_val - first_val) / first_val) * 100
            industry_trends[industry] = {
                'first_period': ind_data.iloc[0]['Period'],
                'last_period': ind_data.iloc[-1]['Period'],
                'first_value': float(first_val),
                'last_value': float(last_val),
                'absolute_change': float(last_val - first_val),
                'percent_change': float(pct_change)
            }
            print(f"\n{industry}:")
            print(f"  {ind_data.iloc[0]['Period']}: {first_val:,.0f} → {ind_data.iloc[-1]['Period']}: {last_val:,.0f}")
            print(f"  Change: {last_val - first_val:+,.0f} ({pct_change:+.2f}%)")

# Age Group Trends
print("\n" + "=" * 80)
print("AGE GROUP EMPLOYMENT TRENDS")
print("=" * 80)

age_filled = age_df[
    (age_df['Series_title_1'] == 'Filled jobs') & 
    (age_df['Series_title_3'] == 'Actual')
].copy()

age_trends = {}
for age_group in sorted(age_filled['Series_title_2'].unique()):
    age_data = age_filled[age_filled['Series_title_2'] == age_group].sort_values('Period')
    if len(age_data) > 0:
        first_val = age_data.iloc[0]['Data_value_numeric']
        last_val = age_data.iloc[-1]['Data_value_numeric']
        if pd.notna(first_val) and pd.notna(last_val) and first_val > 0:
            pct_change = ((last_val - first_val) / first_val) * 100
            age_trends[age_group] = {
                'first_value': float(first_val),
                'last_value': float(last_val),
                'percent_change': float(pct_change)
            }
            print(f"\n{age_group}:")
            print(f"  {age_data.iloc[0]['Period']}: {first_val:,.0f} → {age_data.iloc[-1]['Period']}: {last_val:,.0f}")
            print(f"  Change: {pct_change:+.2f}%")

# Gender Analysis
print("\n" + "=" * 80)
print("GENDER EMPLOYMENT TRENDS")
print("=" * 80)

sex_filled = sex_df[
    (sex_df['Series_title_1'] == 'Filled jobs') & 
    (sex_df['Series_title_3'] == 'Actual')
].copy()

for gender in sorted(sex_filled['Series_title_2'].unique()):
    gender_data = sex_filled[sex_filled['Series_title_2'] == gender].sort_values('Period')
    if len(gender_data) > 0:
        first_val = gender_data.iloc[0]['Data_value_numeric']
        last_val = gender_data.iloc[-1]['Data_value_numeric']
        if pd.notna(first_val) and pd.notna(last_val):
            pct_change = ((last_val - first_val) / first_val) * 100
            print(f"\n{gender}:")
            print(f"  {gender_data.iloc[0]['Period']}: {first_val:,.0f} → {gender_data.iloc[-1]['Period']}: {last_val:,.0f}")
            print(f"  Change: {last_val - first_val:+,.0f} ({pct_change:+.2f}%)")

# COVID-19 Impact Analysis (2019-2021)
print("\n" + "=" * 80)
print("COVID-19 IMPACT ANALYSIS (2019-2021)")
print("=" * 80)

covid_period = df[(df['Year'] >= 2019) & (df['Year'] <= 2021) & (df['Series_title_3'] == 'Actual')]
pre_covid = df[(df['Period'] == 2019.12) & (df['Series_title_3'] == 'Actual')]
during_covid = df[(df['Period'] == 2020.06) & (df['Series_title_3'] == 'Actual')]

print("\nComparing Q4 2019 (pre-COVID) vs Q2 2020 (during COVID):")
print(f"  Pre-COVID records: {len(pre_covid)}")
print(f"  During COVID records: {len(during_covid)}")

# Save Phase 3 findings
phase3_findings = {
    'industry_trends': industry_trends,
    'age_trends': age_trends,
    'analysis_date': datetime.now().isoformat(),
    'key_insights': {
        'total_industries': len(industry_trends),
        'total_age_groups': len(age_trends),
        'data_span_years': 15,
        'quarterly_data': True
    }
}

with open('/vercel/sandbox/phase3_findings.json', 'w') as f:
    json.dump(phase3_findings, f, indent=2)

print("\n" + "=" * 80)
print("✓ Phase 3 Complete - Findings saved to phase3_findings.json")
print("=" * 80)
