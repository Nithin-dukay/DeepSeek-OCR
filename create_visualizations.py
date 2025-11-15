import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (14, 8)

# Read data
print("Loading data...")
df = pd.read_csv('/vercel/sandbox/uploads/employment-data.csv')

# Convert Period to datetime for better plotting
df['Year'] = df['Period'].apply(lambda x: int(x) if pd.notna(x) else None)
df['Quarter'] = df['Period'].apply(lambda x: int((x % 1) * 100 / 3) + 1 if pd.notna(x) else None)
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                             ((df['Quarter'] - 1) * 3 + 1).astype(str) + '-01', 
                             errors='coerce')

# Create output directory for images
import os
os.makedirs('/vercel/sandbox/visualizations', exist_ok=True)

print("Creating visualizations...")

# ============================================================================
# VISUALIZATION 1: Employment by Industry Over Time (Seasonally Adjusted)
# ============================================================================
print("1. Industry employment trends...")
fig, ax = plt.subplots(figsize=(16, 10))

industry_df = df[(df['Series_title_1'] == 'Filled jobs') & 
                  (df['Series_title_3'] == 'Seasonally adjusted') &
                  (df['Group'] == 'Industry by employment variable')]

industries = ['Agriculture, Forestry and Fishing', 'Mining', 'Manufacturing', 
              'Electricity, Gas, Water and Waste Services', 'Health Care and Social Assistance',
              'Education and Training', 'Arts and Recreation Services', 'Other Services']

for industry in industries:
    ind_data = industry_df[industry_df['Series_title_2'] == industry].sort_values('Period')
    if len(ind_data) > 0:
        ax.plot(ind_data['Date'], ind_data['Data_value'], marker='o', 
                linewidth=2, markersize=4, label=industry, alpha=0.8)

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Filled Jobs', fontsize=14, fontweight='bold')
ax.set_title('Employment Trends by Industry (2011-2025)\nSeasonally Adjusted', 
             fontsize=18, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/01_industry_trends.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 2: Total Industry Employment - Actual vs Seasonally Adjusted
# ============================================================================
print("2. Total industry comparison...")
fig, ax = plt.subplots(figsize=(16, 8))

total_ind_df = df[(df['Series_title_1'] == 'Filled jobs') & 
                   (df['Series_title_2'] == 'Total Industry') &
                   (df['Group'] == 'Industry by employment variable')]

for measure_type in ['Actual', 'Seasonally adjusted', 'Trend']:
    data = total_ind_df[total_ind_df['Series_title_3'] == measure_type].sort_values('Period')
    if len(data) > 0:
        ax.plot(data['Date'], data['Data_value'], marker='o', 
                linewidth=2.5, markersize=5, label=measure_type, alpha=0.8)

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Filled Jobs', fontsize=14, fontweight='bold')
ax.set_title('Total Industry Employment: Actual vs Seasonally Adjusted vs Trend (2011-2025)', 
             fontsize=18, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/02_total_employment_comparison.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 3: Employment by Age Group
# ============================================================================
print("3. Age group employment...")
fig, ax = plt.subplots(figsize=(16, 10))

age_df = df[(df['Group'] == 'Age by employment variable') & 
            (df['Series_title_1'] == 'Filled jobs') &
            (df['Series_title_3'] == 'Actual')]

age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', 
              '45-49', '50-54', '55-59', '60-64', '65 +']

for age in age_groups:
    age_data = age_df[age_df['Series_title_2'] == age].sort_values('Period')
    if len(age_data) > 0:
        ax.plot(age_data['Date'], age_data['Data_value'], marker='o', 
                linewidth=2, markersize=4, label=age, alpha=0.8)

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Filled Jobs', fontsize=14, fontweight='bold')
ax.set_title('Employment Trends by Age Group (2011-2025)', 
             fontsize=18, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/03_age_group_trends.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 4: Employment by Sex
# ============================================================================
print("4. Gender employment comparison...")
fig, ax = plt.subplots(figsize=(16, 8))

sex_jobs_df = df[(df['Group'] == 'Sex by employment variable') & 
                  (df['Series_title_1'] == 'Filled jobs') &
                  (df['Series_title_3'] == 'Actual')]

for sex in ['Male', 'Female']:
    sex_data = sex_jobs_df[sex_jobs_df['Series_title_2'] == sex].sort_values('Period')
    if len(sex_data) > 0:
        ax.plot(sex_data['Date'], sex_data['Data_value'], marker='o', 
                linewidth=3, markersize=6, label=sex, alpha=0.8)

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Filled Jobs', fontsize=14, fontweight='bold')
ax.set_title('Employment Trends by Gender (2011-2025)', 
             fontsize=18, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/04_gender_employment.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 5: Industry Earnings Growth
# ============================================================================
print("5. Industry earnings trends...")
fig, ax = plt.subplots(figsize=(16, 10))

earnings_df = df[(df['Series_title_1'] == 'Total earnings') & 
                  (df['Series_title_3'] == 'Actual') &
                  (df['Group'] == 'Industry by employment variable')]

industries_earnings = ['Health Care and Social Assistance', 'Education and Training',
                       'Other Services', 'Arts and Recreation Services']

for industry in industries_earnings:
    ind_data = earnings_df[earnings_df['Series_title_2'] == industry].sort_values('Period')
    if len(ind_data) > 0:
        ax.plot(ind_data['Date'], ind_data['Data_value'], marker='o', 
                linewidth=2.5, markersize=5, label=industry, alpha=0.8)

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Earnings (Millions $)', fontsize=14, fontweight='bold')
ax.set_title('Total Earnings by Industry (2011-2025)', 
             fontsize=18, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/05_industry_earnings.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 6: Latest Employment by Industry (Bar Chart)
# ============================================================================
print("6. Latest employment by industry...")
fig, ax = plt.subplots(figsize=(14, 10))

latest_period = industry_df['Period'].max()
latest_data = industry_df[(industry_df['Period'] == latest_period) & 
                           (industry_df['Series_title_2'] != 'Total Industry')]

if len(latest_data) > 0:
    latest_sorted = latest_data.sort_values('Data_value', ascending=True)
    colors = sns.color_palette("viridis", len(latest_sorted))
    
    bars = ax.barh(latest_sorted['Series_title_2'], latest_sorted['Data_value'], color=colors)
    
    # Add value labels
    for i, (idx, row) in enumerate(latest_sorted.iterrows()):
        ax.text(row['Data_value'], i, f" {row['Data_value']:,.0f}", 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Filled Jobs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Industry', fontsize=14, fontweight='bold')
    ax.set_title(f'Employment by Industry (Latest: Q2 2025)\nSeasonally Adjusted', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/06_latest_industry_employment.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 7: Age Distribution Heatmap
# ============================================================================
print("7. Age distribution heatmap...")
fig, ax = plt.subplots(figsize=(16, 10))

age_pivot = age_df.pivot_table(values='Data_value', 
                                index='Series_title_2', 
                                columns='Period', 
                                aggfunc='mean')

# Sort age groups properly
age_order = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', 
             '45-49', '50-54', '55-59', '60-64', '65 +']
age_pivot = age_pivot.reindex([a for a in age_order if a in age_pivot.index])

sns.heatmap(age_pivot, cmap='YlOrRd', annot=False, fmt='.0f', 
            cbar_kws={'label': 'Number of Jobs'}, ax=ax, linewidths=0.5)

ax.set_xlabel('Period', fontsize=14, fontweight='bold')
ax.set_ylabel('Age Group', fontsize=14, fontweight='bold')
ax.set_title('Employment Heatmap by Age Group Over Time (2011-2025)', 
             fontsize=18, fontweight='bold', pad=20)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/07_age_heatmap.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 8: Regional Employment Distribution
# ============================================================================
print("8. Regional employment...")
fig, ax = plt.subplots(figsize=(14, 10))

region_df = df[(df['Group'] == 'Region by employment variable') & 
               (df['Series_title_1'] == 'Filled jobs') &
               (df['Series_title_3'] == 'Actual')]

latest_period = region_df['Period'].max()
latest_regional = region_df[region_df['Period'] == latest_period].sort_values('Data_value', ascending=True)

if len(latest_regional) > 0:
    colors = sns.color_palette("coolwarm", len(latest_regional))
    bars = ax.barh(latest_regional['Series_title_2'], latest_regional['Data_value'], color=colors)
    
    for i, (idx, row) in enumerate(latest_regional.iterrows()):
        ax.text(row['Data_value'], i, f" {row['Data_value']:,.0f}", 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Filled Jobs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Region', fontsize=14, fontweight='bold')
    ax.set_title(f'Employment by Region (Latest: Q2 2025)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/08_regional_employment.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 9: Quarterly Employment Patterns (Seasonality)
# ============================================================================
print("9. Quarterly patterns...")
fig, ax = plt.subplots(figsize=(14, 8))

total_actual = df[(df['Series_title_1'] == 'Filled jobs') & 
                   (df['Series_title_2'] == 'Total Industry') &
                   (df['Series_title_3'] == 'Actual') &
                   (df['Group'] == 'Industry by employment variable')].copy()

if len(total_actual) > 0:
    # Group by quarter
    quarter_map = {2011.06: 'Q2', 2011.09: 'Q3', 2011.12: 'Q4'}
    total_actual['Quarter_Label'] = total_actual['Period'].apply(
        lambda x: f"Q{int((x % 1) * 100 / 3) + 1}" if pd.notna(x) else None
    )
    
    quarter_avg = total_actual.groupby('Quarter_Label')['Data_value'].agg(['mean', 'std'])
    quarter_order = ['Q1', 'Q2', 'Q3', 'Q4']
    quarter_avg = quarter_avg.reindex([q for q in quarter_order if q in quarter_avg.index])
    
    bars = ax.bar(quarter_avg.index, quarter_avg['mean'], 
                  yerr=quarter_avg['std'], capsize=10, 
                  color=sns.color_palette("Set2", 4), alpha=0.8, edgecolor='black', linewidth=2)
    
    for i, (idx, row) in enumerate(quarter_avg.iterrows()):
        ax.text(i, row['mean'], f"{row['mean']:,.0f}", 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Quarter', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Filled Jobs', fontsize=14, fontweight='bold')
    ax.set_title('Seasonal Employment Patterns - Total Industry\nAverage by Quarter (2011-2025)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/09_seasonal_patterns.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 10: Gender Employment Comparison with Gap
# ============================================================================
print("10. Gender gap analysis...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

sex_df = df[(df['Group'] == 'Sex by employment variable') & 
            (df['Series_title_1'] == 'Filled jobs') &
            (df['Series_title_3'] == 'Actual')]

male_data = sex_df[sex_df['Series_title_2'] == 'Male'].sort_values('Period')
female_data = sex_df[sex_df['Series_title_2'] == 'Female'].sort_values('Period')

# Plot 1: Absolute numbers
ax1.plot(male_data['Date'], male_data['Data_value'], marker='o', 
         linewidth=3, markersize=6, label='Male', color='#3498db', alpha=0.8)
ax1.plot(female_data['Date'], female_data['Data_value'], marker='s', 
         linewidth=3, markersize=6, label='Female', color='#e74c3c', alpha=0.8)
ax1.fill_between(male_data['Date'], male_data['Data_value'], alpha=0.2, color='#3498db')
ax1.fill_between(female_data['Date'], female_data['Data_value'], alpha=0.2, color='#e74c3c')

ax1.set_ylabel('Number of Filled Jobs', fontsize=13, fontweight='bold')
ax1.set_title('Employment by Gender (2011-2025)', fontsize=16, fontweight='bold', pad=15)
ax1.legend(loc='best', fontsize=12, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Plot 2: Gender gap
if len(male_data) == len(female_data):
    gap = female_data['Data_value'].values - male_data['Data_value'].values
    colors = ['#2ecc71' if g >= 0 else '#e67e22' for g in gap]
    ax2.bar(male_data['Date'], gap, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
    
    ax2.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Employment Gap (Female - Male)', fontsize=13, fontweight='bold')
    ax2.set_title('Gender Employment Gap Over Time', fontsize=16, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/10_gender_analysis.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 11: Industry Growth Rates
# ============================================================================
print("11. Industry growth rates...")
fig, ax = plt.subplots(figsize=(14, 10))

growth_data = []
for industry in industries:
    ind_data = industry_df[industry_df['Series_title_2'] == industry].sort_values('Period')
    if len(ind_data) > 1:
        first_val = ind_data['Data_value'].iloc[0]
        last_val = ind_data['Data_value'].iloc[-1]
        growth_pct = ((last_val - first_val) / first_val * 100)
        growth_data.append({'Industry': industry, 'Growth': growth_pct})

growth_df = pd.DataFrame(growth_data).sort_values('Growth', ascending=True)
colors = ['#e74c3c' if g < 0 else '#2ecc71' for g in growth_df['Growth']]

bars = ax.barh(growth_df['Industry'], growth_df['Growth'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (idx, row) in enumerate(growth_df.iterrows()):
    ax.text(row['Growth'], i, f" {row['Growth']:+.1f}%", 
            va='center', fontsize=11, fontweight='bold')

ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_xlabel('Growth Rate (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Industry', fontsize=14, fontweight='bold')
ax.set_title('Employment Growth by Industry (2011-2025)\nSeasonally Adjusted', 
             fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/11_industry_growth.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 12: Age Group Distribution (Latest Period)
# ============================================================================
print("12. Age distribution pie chart...")
fig, ax = plt.subplots(figsize=(14, 10))

latest_age = age_df[age_df['Period'] == age_df['Period'].max()]
age_values = []
age_labels = []

for age in age_groups:
    val = latest_age[latest_age['Series_title_2'] == age]['Data_value'].values
    if len(val) > 0:
        age_values.append(val[0])
        age_labels.append(age)

colors = sns.color_palette("Set3", len(age_values))
wedges, texts, autotexts = ax.pie(age_values, labels=age_labels, autopct='%1.1f%%',
                                    colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})

for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(10)

ax.set_title('Employment Distribution by Age Group (Q2 2025)', 
             fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/12_age_distribution.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 13: Multi-panel Dashboard - Industry Overview
# ============================================================================
print("13. Industry dashboard...")
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Panel 1: Top industries by employment
ax1 = fig.add_subplot(gs[0, 0])
top_industries = latest_data.nlargest(5, 'Data_value')
ax1.bar(range(len(top_industries)), top_industries['Data_value'], 
        color=sns.color_palette("rocket", 5), alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xticks(range(len(top_industries)))
ax1.set_xticklabels([ind[:30] for ind in top_industries['Series_title_2']], rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Filled Jobs', fontsize=11, fontweight='bold')
ax1.set_title('Top 5 Industries by Employment (Q2 2025)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Employment trend - Manufacturing
ax2 = fig.add_subplot(gs[0, 1])
mfg_data = industry_df[industry_df['Series_title_2'] == 'Manufacturing'].sort_values('Period')
ax2.plot(mfg_data['Date'], mfg_data['Data_value'], marker='o', 
         linewidth=2.5, markersize=5, color='#9b59b6', alpha=0.8)
ax2.fill_between(mfg_data['Date'], mfg_data['Data_value'], alpha=0.3, color='#9b59b6')
ax2.set_ylabel('Filled Jobs', fontsize=11, fontweight='bold')
ax2.set_title('Manufacturing Employment Trend', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# Panel 3: Healthcare trend
ax3 = fig.add_subplot(gs[1, 0])
health_earnings = earnings_df[earnings_df['Series_title_2'] == 'Health Care and Social Assistance'].sort_values('Period')
ax3.plot(health_earnings['Date'], health_earnings['Data_value'], marker='o', 
         linewidth=2.5, markersize=5, color='#e74c3c', alpha=0.8)
ax3.fill_between(health_earnings['Date'], health_earnings['Data_value'], alpha=0.3, color='#e74c3c')
ax3.set_ylabel('Total Earnings ($M)', fontsize=11, fontweight='bold')
ax3.set_title('Healthcare Earnings Trend', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# Panel 4: Education earnings trend
ax4 = fig.add_subplot(gs[1, 1])
edu_earnings = earnings_df[earnings_df['Series_title_2'] == 'Education and Training'].sort_values('Period')
ax4.plot(edu_earnings['Date'], edu_earnings['Data_value'], marker='o', 
         linewidth=2.5, markersize=5, color='#3498db', alpha=0.8)
ax4.fill_between(edu_earnings['Date'], edu_earnings['Data_value'], alpha=0.3, color='#3498db')
ax4.set_ylabel('Total Earnings ($M)', fontsize=11, fontweight='bold')
ax4.set_title('Education Earnings Trend', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

# Panel 5: Age group comparison (latest)
ax5 = fig.add_subplot(gs[2, :])
age_latest = age_df[age_df['Period'] == age_df['Period'].max()]
age_sorted = []
for age in age_groups:
    val = age_latest[age_latest['Series_title_2'] == age]['Data_value'].values
    if len(val) > 0:
        age_sorted.append({'Age': age, 'Jobs': val[0]})

age_sorted_df = pd.DataFrame(age_sorted)
colors = sns.color_palette("viridis", len(age_sorted_df))
bars = ax5.bar(age_sorted_df['Age'], age_sorted_df['Jobs'], 
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, row in age_sorted_df.iterrows():
    ax5.text(i, row['Jobs'], f"{row['Jobs']:,.0f}", 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax5.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax5.set_ylabel('Filled Jobs', fontsize=12, fontweight='bold')
ax5.set_title('Employment by Age Group (Q2 2025)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

plt.suptitle('Employment Data Dashboard - New Zealand', fontsize=20, fontweight='bold', y=0.995)
plt.savefig('/vercel/sandbox/visualizations/13_industry_dashboard.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 14: Year-over-Year Growth Analysis
# ============================================================================
print("14. YoY growth analysis...")
fig, ax = plt.subplots(figsize=(16, 8))

total_sa = df[(df['Series_title_1'] == 'Filled jobs') & 
               (df['Series_title_2'] == 'Total Industry') &
               (df['Series_title_3'] == 'Seasonally adjusted') &
               (df['Group'] == 'Industry by employment variable')].sort_values('Period')

if len(total_sa) > 4:
    # Calculate YoY growth (4 quarters)
    yoy_growth = []
    for i in range(4, len(total_sa)):
        current = total_sa.iloc[i]['Data_value']
        previous = total_sa.iloc[i-4]['Data_value']
        if pd.notna(current) and pd.notna(previous) and previous > 0:
            growth = ((current - previous) / previous * 100)
            yoy_growth.append({
                'Date': total_sa.iloc[i]['Date'],
                'Growth': growth
            })
    
    yoy_df = pd.DataFrame(yoy_growth)
    colors = ['#2ecc71' if g >= 0 else '#e74c3c' for g in yoy_df['Growth']]
    
    ax.bar(yoy_df['Date'], yoy_df['Growth'], color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Year-over-Year Growth (%)', fontsize=14, fontweight='bold')
    ax.set_title('Total Employment Year-over-Year Growth Rate (2012-2025)\nSeasonally Adjusted', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/14_yoy_growth.png', bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 15: Correlation Matrix - Industries
# ============================================================================
print("15. Industry correlation matrix...")
fig, ax = plt.subplots(figsize=(14, 12))

# Create pivot table for correlation
industry_pivot = industry_df.pivot_table(values='Data_value', 
                                          index='Period', 
                                          columns='Series_title_2', 
                                          aggfunc='mean')

# Select main industries
main_industries = ['Agriculture, Forestry and Fishing', 'Mining', 'Manufacturing',
                   'Health Care and Social Assistance', 'Education and Training']
industry_pivot_subset = industry_pivot[[col for col in main_industries if col in industry_pivot.columns]]

# Calculate correlation
corr_matrix = industry_pivot_subset.corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            ax=ax, vmin=-1, vmax=1)

ax.set_title('Industry Employment Correlation Matrix\n(Seasonally Adjusted)', 
             fontsize=18, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig('/vercel/sandbox/visualizations/15_correlation_matrix.png', bbox_inches='tight')
plt.close()

print("\nAll visualizations created successfully!")
print(f"Saved to: /vercel/sandbox/visualizations/")
