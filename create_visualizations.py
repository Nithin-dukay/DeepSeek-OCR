#!/usr/bin/env python3
"""
Employment Data Visualization Script - Creates comprehensive PDF report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("CREATING COMPREHENSIVE VISUALIZATION REPORT")
print("=" * 80)

# Load data
df = pd.read_csv('/vercel/sandbox/uploads/employment-data.csv')
df['Data_value_numeric'] = pd.to_numeric(df['Data_value'], errors='coerce')
df['Year'] = df['Period'].apply(lambda x: int(x) if pd.notna(x) else None)
df['Quarter'] = df['Period'].apply(lambda x: int((x % 1) * 100) if pd.notna(x) else None)

print(f"\n✓ Data loaded: {len(df):,} records")

# Create PDF
pdf_path = '/vercel/sandbox/employment_analysis_report.pdf'
pdf = PdfPages(pdf_path)

# Chart 1: Industry Employment Trends Over Time
print("\n📊 Creating Chart 1: Industry Employment Trends...")
fig, ax = plt.subplots(figsize=(16, 10))

industry_df = df[
    (df['Group'] == 'Industry by employment variable') &
    (df['Series_title_1'] == 'Filled jobs') & 
    (df['Series_title_3'] == 'Actual')
].copy()

industries_to_plot = ['Agriculture, Forestry and Fishing', 'Manufacturing', 
                      'Mining', 'Electricity, Gas, Water and Waste Services']

for industry in industries_to_plot:
    ind_data = industry_df[industry_df['Series_title_2'] == industry].sort_values('Period')
    if len(ind_data) > 0:
        ax.plot(ind_data['Period'], ind_data['Data_value_numeric'], 
                marker='o', linewidth=2, markersize=4, label=industry, alpha=0.8)

ax.set_xlabel('Period (Year.Quarter)', fontsize=14, fontweight='bold')
ax.set_ylabel('Filled Jobs', fontsize=14, fontweight='bold')
ax.set_title('Industry Employment Trends (2011-2025)', fontsize=18, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 2: Age Group Distribution Over Time
print("📊 Creating Chart 2: Age Group Employment Trends...")
fig, ax = plt.subplots(figsize=(16, 10))

age_df = df[
    (df['Group'] == 'Age by employment variable') &
    (df['Series_title_1'] == 'Filled jobs') & 
    (df['Series_title_3'] == 'Actual')
].copy()

age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65 +']

for age_group in age_groups:
    age_data = age_df[age_df['Series_title_2'] == age_group].sort_values('Period')
    if len(age_data) > 0:
        ax.plot(age_data['Period'], age_data['Data_value_numeric'], 
                marker='o', linewidth=2, markersize=3, label=age_group, alpha=0.7)

ax.set_xlabel('Period (Year.Quarter)', fontsize=14, fontweight='bold')
ax.set_ylabel('Filled Jobs', fontsize=14, fontweight='bold')
ax.set_title('Employment by Age Group (2011-2025)', fontsize=18, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 3: Gender Employment Comparison
print("📊 Creating Chart 3: Gender Employment Comparison...")
fig, ax = plt.subplots(figsize=(16, 10))

sex_df = df[
    (df['Group'] == 'Sex by employment variable') &
    (df['Series_title_1'] == 'Filled jobs') & 
    (df['Series_title_3'] == 'Actual')
].copy()

for gender in ['Male', 'Female']:
    gender_data = sex_df[sex_df['Series_title_2'] == gender].sort_values('Period')
    if len(gender_data) > 0:
        ax.plot(gender_data['Period'], gender_data['Data_value_numeric'], 
                marker='o', linewidth=3, markersize=6, label=gender, alpha=0.8)

ax.set_xlabel('Period (Year.Quarter)', fontsize=14, fontweight='bold')
ax.set_ylabel('Filled Jobs', fontsize=14, fontweight='bold')
ax.set_title('Employment by Gender (2011-2025)', fontsize=18, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 4: Regional Employment Comparison (Top 10 Regions)
print("📊 Creating Chart 4: Regional Employment Comparison...")
fig, ax = plt.subplots(figsize=(16, 10))

region_df = df[
    (df['Group'] == 'Region by employment variable') &
    (df['Series_title_1'] == 'Filled jobs') & 
    (df['Series_title_3'] == 'Actual')
].copy()

# Get latest values for each region
latest_period = region_df['Period'].max()
latest_regional = region_df[region_df['Period'] == latest_period].sort_values('Data_value_numeric', ascending=False)

top_10_regions = latest_regional.head(10)
regions = top_10_regions['Series_title_2'].tolist()
values = top_10_regions['Data_value_numeric'].tolist()

bars = ax.barh(range(len(regions)), values, color=sns.color_palette('viridis', len(regions)))
ax.set_yticks(range(len(regions)))
ax.set_yticklabels(regions)
ax.set_xlabel('Filled Jobs', fontsize=14, fontweight='bold')
ax.set_title(f'Top 10 Regions by Employment (Period {latest_period})', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val, i, f' {val:,.0f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 5: Industry Growth Comparison (First vs Last Period)
print("📊 Creating Chart 5: Industry Growth Comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

industry_growth_data = []
for industry in industry_df['Series_title_2'].unique():
    ind_data = industry_df[industry_df['Series_title_2'] == industry].sort_values('Period')
    if len(ind_data) >= 2:
        first_val = ind_data.iloc[0]['Data_value_numeric']
        last_val = ind_data.iloc[-1]['Data_value_numeric']
        if pd.notna(first_val) and pd.notna(last_val) and first_val > 0:
            pct_change = ((last_val - first_val) / first_val) * 100
            industry_growth_data.append({
                'industry': industry,
                'first': first_val,
                'last': last_val,
                'growth': pct_change
            })

industry_growth_df = pd.DataFrame(industry_growth_data).sort_values('growth', ascending=True)

# Growth percentage chart
colors = ['red' if x < 0 else 'green' for x in industry_growth_df['growth']]
ax1.barh(range(len(industry_growth_df)), industry_growth_df['growth'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(industry_growth_df)))
ax1.set_yticklabels(industry_growth_df['industry'], fontsize=9)
ax1.set_xlabel('Growth (%)', fontsize=12, fontweight='bold')
ax1.set_title('Industry Employment Growth (2011-2025)', fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(True, alpha=0.3, axis='x')

# Add percentage labels
for i, val in enumerate(industry_growth_df['growth']):
    ax1.text(val, i, f' {val:.1f}%', va='center', fontsize=9, fontweight='bold')

# Absolute values comparison
x = np.arange(len(industry_growth_df))
width = 0.35
ax2.barh(x - width/2, industry_growth_df['first'], width, label='2011', alpha=0.8, color='steelblue')
ax2.barh(x + width/2, industry_growth_df['last'], width, label='2025', alpha=0.8, color='coral')
ax2.set_yticks(x)
ax2.set_yticklabels(industry_growth_df['industry'], fontsize=9)
ax2.set_xlabel('Filled Jobs', fontsize=12, fontweight='bold')
ax2.set_title('Industry Employment: 2011 vs 2025', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 6: Age Group Distribution (Latest Period)
print("📊 Creating Chart 6: Age Group Distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

latest_age = age_df[
    (age_df['Period'] == age_df['Period'].max()) &
    (age_df['Series_title_3'] == 'Actual')
].sort_values('Data_value_numeric', ascending=False)

# Bar chart
ax1.bar(range(len(latest_age)), latest_age['Data_value_numeric'], 
        color=sns.color_palette('coolwarm', len(latest_age)), alpha=0.8)
ax1.set_xticks(range(len(latest_age)))
ax1.set_xticklabels(latest_age['Series_title_2'], rotation=45, ha='right')
ax1.set_ylabel('Filled Jobs', fontsize=12, fontweight='bold')
ax1.set_title(f'Employment by Age Group (Period {latest_age.iloc[0]["Period"]})', 
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, val in enumerate(latest_age['Data_value_numeric']):
    ax1.text(i, val, f'{val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Pie chart
ax2.pie(latest_age['Data_value_numeric'], labels=latest_age['Series_title_2'], 
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
ax2.set_title('Age Group Distribution (%)', fontsize=14, fontweight='bold')

plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 7: Quarterly Seasonality Patterns
print("📊 Creating Chart 7: Quarterly Seasonality...")
fig, ax = plt.subplots(figsize=(16, 10))

actual_data = df[df['Series_title_3'] == 'Actual'].copy()
quarterly_avg = actual_data.groupby('Quarter')['Data_value_numeric'].mean()

quarters_map = {2: 'Q1 (Feb)', 5: 'Q2 (May)', 8: 'Q3 (Aug)', 11: 'Q4 (Nov)'}
quarter_labels = [quarters_map.get(q, f'Q{q}') for q in sorted(quarterly_avg.index)]

bars = ax.bar(range(len(quarterly_avg)), quarterly_avg.values, 
              color=sns.color_palette('Set2', len(quarterly_avg)), alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(quarterly_avg)))
ax.set_xticklabels(quarter_labels, fontsize=12)
ax.set_ylabel('Average Value', fontsize=14, fontweight='bold')
ax.set_title('Quarterly Seasonality Patterns (Average Values)', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, quarterly_avg.values)):
    ax.text(i, val, f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 8: Regional Employment Map (Heatmap style)
print("📊 Creating Chart 8: Regional Employment Heatmap...")
fig, ax = plt.subplots(figsize=(16, 12))

region_pivot = region_df[
    (region_df['Series_title_1'] == 'Filled jobs') & 
    (region_df['Series_title_3'] == 'Actual')
].pivot_table(
    index='Series_title_2', 
    columns='Year', 
    values='Data_value_numeric', 
    aggfunc='mean'
)

sns.heatmap(region_pivot, annot=False, fmt='.0f', cmap='YlOrRd', 
            cbar_kws={'label': 'Filled Jobs'}, ax=ax, linewidths=0.5)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Region', fontsize=14, fontweight='bold')
ax.set_title('Regional Employment Heatmap (2011-2025)', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 9: Year-over-Year Total Employment
print("📊 Creating Chart 9: Year-over-Year Total Employment...")
fig, ax = plt.subplots(figsize=(16, 10))

yearly_totals = df[
    (df['Series_title_1'] == 'Filled jobs') & 
    (df['Series_title_3'] == 'Actual')
].groupby('Year')['Data_value_numeric'].sum()

ax.plot(yearly_totals.index, yearly_totals.values, marker='o', linewidth=3, 
        markersize=10, color='darkblue', alpha=0.7)
ax.fill_between(yearly_totals.index, yearly_totals.values, alpha=0.3, color='skyblue')

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Filled Jobs', fontsize=14, fontweight='bold')
ax.set_title('Total Employment Growth (2011-2025)', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

# Add value labels
for year, val in zip(yearly_totals.index, yearly_totals.values):
    if pd.notna(year):
        ax.text(year, val, f'{val/1e6:.1f}M', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')

plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 10: Gender Employment Gap Analysis
print("📊 Creating Chart 10: Gender Employment Gap...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

sex_df_actual = df[
    (df['Group'] == 'Sex by employment variable') &
    (df['Series_title_1'] == 'Filled jobs') & 
    (df['Series_title_3'] == 'Actual')
].copy()

male_data = sex_df_actual[sex_df_actual['Series_title_2'] == 'Male'].sort_values('Period')
female_data = sex_df_actual[sex_df_actual['Series_title_2'] == 'Female'].sort_values('Period')

# Absolute values
ax1.plot(male_data['Period'], male_data['Data_value_numeric'], 
         marker='o', linewidth=3, markersize=6, label='Male', color='steelblue', alpha=0.8)
ax1.plot(female_data['Period'], female_data['Data_value_numeric'], 
         marker='s', linewidth=3, markersize=6, label='Female', color='coral', alpha=0.8)
ax1.set_ylabel('Filled Jobs', fontsize=12, fontweight='bold')
ax1.set_title('Gender Employment Trends', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Gap analysis
if len(male_data) == len(female_data):
    gap = male_data['Data_value_numeric'].values - female_data['Data_value_numeric'].values
    ax2.plot(male_data['Period'], gap, marker='o', linewidth=3, markersize=6, 
             color='purple', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(male_data['Period'], gap, 0, alpha=0.3, 
                     color=['red' if x < 0 else 'green' for x in gap])
    ax2.set_xlabel('Period (Year.Quarter)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Employment Gap (Male - Female)', fontsize=12, fontweight='bold')
    ax2.set_title('Gender Employment Gap Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 11: Data Completeness Over Time
print("📊 Creating Chart 11: Data Quality Metrics...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Records per year
records_per_year = df.groupby('Year').size()
ax1.bar(records_per_year.index, records_per_year.values, 
        color=sns.color_palette('muted', len(records_per_year)), alpha=0.8, edgecolor='black')
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
ax1.set_title('Data Records per Year', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for year, val in zip(records_per_year.index, records_per_year.values):
    if pd.notna(year):
        ax1.text(year, val, f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# STATUS distribution
status_counts = df['STATUS'].value_counts()
colors_status = {'F': 'green', 'R': 'orange', 'C': 'red'}
colors = [colors_status.get(s, 'gray') for s in status_counts.index]

wedges, texts, autotexts = ax2.pie(status_counts.values, labels=status_counts.index, 
                                     autopct='%1.1f%%', startangle=90, colors=colors,
                                     textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Data STATUS Distribution\n(F=Final, R=Revised, C=Confidential)', 
              fontsize=14, fontweight='bold')

plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Chart 12: Multi-panel Dashboard - Key Metrics
print("📊 Creating Chart 12: Key Metrics Dashboard...")
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Total Employment by Group
ax1 = fig.add_subplot(gs[0, :])
group_counts = df.groupby('Group').size().sort_values(ascending=False)
ax1.bar(range(len(group_counts)), group_counts.values, 
        color=sns.color_palette('Set3', len(group_counts)), alpha=0.8, edgecolor='black')
ax1.set_xticks(range(len(group_counts)))
ax1.set_xticklabels(group_counts.index, rotation=15, ha='right', fontsize=9)
ax1.set_ylabel('Records', fontsize=11, fontweight='bold')
ax1.set_title('Records by Category Group', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Employment Variable Distribution
ax2 = fig.add_subplot(gs[1, 0])
emp_var_counts = df['Series_title_1'].value_counts()
ax2.pie(emp_var_counts.values, labels=emp_var_counts.index, autopct='%1.1f%%', 
        startangle=90, textprops={'fontsize': 8})
ax2.set_title('Employment Variables', fontsize=11, fontweight='bold')

# Panel 3: Adjustment Type Distribution
ax3 = fig.add_subplot(gs[1, 1])
adj_counts = df['Series_title_3'].value_counts()
ax3.pie(adj_counts.values, labels=adj_counts.index, autopct='%1.1f%%', 
        startangle=90, textprops={'fontsize': 8})
ax3.set_title('Adjustment Types', fontsize=11, fontweight='bold')

# Panel 4: UNITS Distribution
ax4 = fig.add_subplot(gs[1, 2])
units_counts = df['UNITS'].value_counts()
ax4.pie(units_counts.values, labels=units_counts.index, autopct='%1.1f%%', 
        startangle=90, textprops={'fontsize': 8})
ax4.set_title('Units of Measurement', fontsize=11, fontweight='bold')

# Panel 5: Data Completeness
ax5 = fig.add_subplot(gs[2, :])
completeness_data = {
    'Complete Data': df['Data_value_numeric'].notna().sum(),
    'Suppressed Data': (df['Suppressed'] == 'Y').sum(),
    'Missing Data': df['Data_value_numeric'].isna().sum() - (df['Suppressed'] == 'Y').sum()
}
colors_comp = ['green', 'orange', 'red']
bars = ax5.bar(completeness_data.keys(), completeness_data.values(), 
               color=colors_comp, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Number of Records', fontsize=11, fontweight='bold')
ax5.set_title('Data Completeness Overview', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, completeness_data.values()):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:,}\n({val/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

fig.suptitle('Employment Data - Key Metrics Dashboard', fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
pdf.savefig(fig, dpi=300)
plt.close()

# Close PDF
pdf.close()

print(f"\n✓ PDF report created successfully: {pdf_path}")
print(f"  Total charts: 12")
print(f"  File size: {pd.io.common.file_exists(pdf_path)}")

print("\n" + "=" * 80)
print("✓ VISUALIZATION COMPLETE")
print("=" * 80)
