#!/usr/bin/env python3
"""
Employment Data Analysis Script - Phase 4: Comprehensive Statistical Analysis
"""

import pandas as pd
import numpy as np
import json

print("=" * 80)
print("PHASE 4: COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 80)

df = pd.read_csv('/vercel/sandbox/uploads/employment-data.csv')
df['Data_value_numeric'] = pd.to_numeric(df['Data_value'], errors='coerce')
df['Year'] = df['Period'].apply(lambda x: int(x) if pd.notna(x) else None)
df['Quarter'] = df['Period'].apply(lambda x: int((x % 1) * 100) if pd.notna(x) else None)

# Regional Employment Analysis
print("\n" + "=" * 80)
print("REGIONAL EMPLOYMENT ANALYSIS")
print("=" * 80)

region_df = df[df['Group'] == 'Region by employment variable']
region_filled = region_df[
    (region_df['Series_title_1'] == 'Filled jobs') & 
    (region_df['Series_title_3'] == 'Actual')
].copy()

print(f"\nRegions analyzed: {region_filled['Series_title_2'].nunique()}")

regional_stats = {}
for region in sorted(region_filled['Series_title_2'].unique()):
    reg_data = region_filled[region_filled['Series_title_2'] == region].sort_values('Period')
    if len(reg_data) > 0:
        first_val = reg_data.iloc[0]['Data_value_numeric']
        last_val = reg_data.iloc[-1]['Data_value_numeric']
        if pd.notna(first_val) and pd.notna(last_val) and first_val > 0:
            pct_change = ((last_val - first_val) / first_val) * 100
            regional_stats[region] = {
                'first_value': float(first_val),
                'last_value': float(last_val),
                'percent_change': float(pct_change),
                'avg_value': float(reg_data['Data_value_numeric'].mean())
            }
            print(f"\n{region}:")
            print(f"  Start: {first_val:,.0f} → End: {last_val:,.0f}")
            print(f"  Growth: {pct_change:+.2f}%")
            print(f"  Average: {reg_data['Data_value_numeric'].mean():,.0f}")

# Quarterly Seasonality Analysis
print("\n" + "=" * 80)
print("QUARTERLY SEASONALITY PATTERNS")
print("=" * 80)

actual_data = df[df['Series_title_3'] == 'Actual'].copy()
quarterly_stats = actual_data.groupby('Quarter')['Data_value_numeric'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std')
]).round(2)

print("\nQuarterly Statistics (all data):")
for quarter in sorted(actual_data['Quarter'].dropna().unique()):
    q_data = actual_data[actual_data['Quarter'] == quarter]['Data_value_numeric']
    print(f"\nQuarter {int(quarter)} (Month {int(quarter)}):")
    print(f"  Records: {len(q_data):,}")
    print(f"  Mean: {q_data.mean():,.2f}")
    print(f"  Median: {q_data.median():,.2f}")
    print(f"  Std Dev: {q_data.std():,.2f}")

# Year-over-Year Growth Analysis
print("\n" + "=" * 80)
print("YEAR-OVER-YEAR GROWTH ANALYSIS")
print("=" * 80)

# Focus on total filled jobs by year
total_jobs_df = df[
    (df['Series_title_1'] == 'Filled jobs') & 
    (df['Series_title_3'] == 'Actual')
].copy()

yearly_totals = total_jobs_df.groupby('Year')['Data_value_numeric'].agg([
    ('total', 'sum'),
    ('count', 'count'),
    ('mean', 'mean')
]).round(0)

print("\nYearly Employment Metrics:")
for year in sorted(yearly_totals.index):
    if pd.notna(year):
        row = yearly_totals.loc[year]
        print(f"\n{int(year)}:")
        print(f"  Total jobs (sum): {row['total']:,.0f}")
        print(f"  Average per record: {row['mean']:,.0f}")
        print(f"  Number of records: {int(row['count'])}")

# Data Quality Analysis
print("\n" + "=" * 80)
print("DATA QUALITY METRICS")
print("=" * 80)

print(f"\nOverall Data Quality:")
print(f"  Total records: {len(df):,}")
print(f"  Complete records (no nulls in Data_value): {df['Data_value_numeric'].notna().sum():,} ({df['Data_value_numeric'].notna().sum()/len(df)*100:.2f}%)")
print(f"  Suppressed records: {(df['Suppressed'] == 'Y').sum():,} ({(df['Suppressed'] == 'Y').sum()/len(df)*100:.2f}%)")
print(f"  Missing data: {df['Data_value_numeric'].isna().sum():,} ({df['Data_value_numeric'].isna().sum()/len(df)*100:.2f}%)")

print("\nSTATUS Code Distribution:")
for status in sorted(df['STATUS'].unique()):
    count = (df['STATUS'] == status).sum()
    print(f"  {status}: {count:,} ({count/len(df)*100:.2f}%)")
    if status == 'F':
        print(f"      (Final)")
    elif status == 'R':
        print(f"      (Revised)")
    elif status == 'C':
        print(f"      (Confidential/Suppressed)")

# Top Growing Industries
print("\n" + "=" * 80)
print("TOP GROWING INDUSTRIES (by % change)")
print("=" * 80)

industry_filled = df[
    (df['Group'] == 'Industry by employment variable') &
    (df['Series_title_1'] == 'Filled jobs') & 
    (df['Series_title_3'] == 'Actual')
].copy()

industry_growth = []
for industry in industry_filled['Series_title_2'].unique():
    ind_data = industry_filled[industry_filled['Series_title_2'] == industry].sort_values('Period')
    if len(ind_data) >= 2:
        first_val = ind_data.iloc[0]['Data_value_numeric']
        last_val = ind_data.iloc[-1]['Data_value_numeric']
        if pd.notna(first_val) and pd.notna(last_val) and first_val > 0:
            pct_change = ((last_val - first_val) / first_val) * 100
            industry_growth.append({
                'industry': industry,
                'growth_pct': pct_change,
                'start_value': first_val,
                'end_value': last_val
            })

industry_growth_sorted = sorted(industry_growth, key=lambda x: x['growth_pct'], reverse=True)
print("\nTop 5 Growing Industries:")
for i, ind in enumerate(industry_growth_sorted[:5], 1):
    print(f"\n{i}. {ind['industry']}")
    print(f"   Growth: {ind['growth_pct']:+.2f}%")
    print(f"   {ind['start_value']:,.0f} → {ind['end_value']:,.0f}")

# Save comprehensive findings
comprehensive_findings = {
    'dataset_summary': {
        'total_records': int(len(df)),
        'date_range': '2011 Q2 to 2025 Q2',
        'years_covered': 15,
        'total_quarters': 57
    },
    'data_categories': {
        'groups': list(df['Group'].unique()),
        'employment_variables': list(df['Series_title_1'].unique()),
        'adjustment_types': list(df['Series_title_3'].unique())
    },
    'regional_statistics': regional_stats,
    'industry_growth_rankings': [
        {
            'industry': ind['industry'],
            'growth_percent': float(ind['growth_pct']),
            'start_value': float(ind['start_value']),
            'end_value': float(ind['end_value'])
        }
        for ind in industry_growth_sorted
    ],
    'data_quality': {
        'completeness_pct': float(df['Data_value_numeric'].notna().sum() / len(df) * 100),
        'suppressed_pct': float((df['Suppressed'] == 'Y').sum() / len(df) * 100),
        'status_distribution': {
            status: int((df['STATUS'] == status).sum())
            for status in df['STATUS'].unique()
        }
    }
}

with open('/vercel/sandbox/comprehensive_findings.json', 'w') as f:
    json.dump(comprehensive_findings, f, indent=2)

print("\n" + "=" * 80)
print("✓ Phase 4 Complete - Comprehensive findings saved")
print("=" * 80)
