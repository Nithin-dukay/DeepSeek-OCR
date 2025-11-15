import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

# Read data
df = pd.read_csv('/vercel/sandbox/uploads/employment-data.csv')

# Create comprehensive summary dashboard
fig = plt.figure(figsize=(20, 14))
fig.suptitle('New Zealand Employment Data - Executive Dashboard (2011-2025)', 
             fontsize=24, fontweight='bold', y=0.98)

# Create grid
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3, top=0.94, bottom=0.05, left=0.05, right=0.95)

# ============================================================================
# Panel 1: Total Employment Over Time
# ============================================================================
ax1 = fig.add_subplot(gs[0, :2])
total_df = df[(df['Series_title_1'] == 'Filled jobs') & 
              (df['Series_title_2'] == 'Total Industry') &
              (df['Series_title_3'] == 'Actual') &
              (df['Group'] == 'Industry by employment variable')].sort_values('Period')

if len(total_df) > 0:
    dates = pd.to_datetime(total_df['Period'].apply(lambda x: f"{int(x)}-{int((x % 1) * 100):02d}-01"))
    ax1.plot(dates, total_df['Data_value'], linewidth=3, color='#2c3e50', marker='o', markersize=4)
    ax1.fill_between(dates, total_df['Data_value'], alpha=0.3, color='#3498db')
    
    # Add annotations for key points
    max_idx = total_df['Data_value'].idxmax()
    min_idx = total_df['Data_value'].idxmin()
    
    ax1.set_ylabel('Total Filled Jobs', fontsize=12, fontweight='bold')
    ax1.set_title('Total Employment Trend', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

# ============================================================================
# Panel 2: Key Metrics Summary
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')

# Calculate key metrics
if len(total_df) > 0:
    latest_employment = total_df['Data_value'].iloc[-1]
    first_employment = total_df['Data_value'].iloc[0]
    growth_pct = ((latest_employment - first_employment) / first_employment * 100)
    
    metrics_text = f"""
KEY METRICS (Q2 2025)

Total Employment:
{latest_employment:,.0f} jobs

Growth (2011-2025):
+{growth_pct:.1f}%

Total Records:
20,108

Time Periods:
57 quarters

Industries:
10 sectors

Regions:
16 regions
"""
    
    ax2.text(0.1, 0.95, metrics_text, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# Panel 3: Top Industries by Employment
# ============================================================================
ax3 = fig.add_subplot(gs[1, :])

industry_df = df[(df['Series_title_1'] == 'Filled jobs') & 
                  (df['Series_title_3'] == 'Seasonally adjusted') &
                  (df['Group'] == 'Industry by employment variable')]

latest_period = industry_df['Period'].max()
latest_ind = industry_df[(industry_df['Period'] == latest_period) & 
                          (industry_df['Series_title_2'] != 'Total Industry')].sort_values('Data_value', ascending=False)

if len(latest_ind) > 0:
    colors = sns.color_palette("rocket_r", len(latest_ind))
    bars = ax3.bar(range(len(latest_ind)), latest_ind['Data_value'], color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_xticks(range(len(latest_ind)))
    ax3.set_xticklabels([ind[:25] for ind in latest_ind['Series_title_2']], 
                        rotation=45, ha='right', fontsize=10)
    
    # Add value labels
    for i, val in enumerate(latest_ind['Data_value']):
        ax3.text(i, val, f'{val:,.0f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    ax3.set_ylabel('Filled Jobs (Seasonally Adjusted)', fontsize=12, fontweight='bold')
    ax3.set_title('Employment by Industry - Q2 2025', fontsize=14, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Panel 4: Gender Employment Comparison
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])

sex_df = df[(df['Group'] == 'Sex by employment variable') & 
            (df['Series_title_1'] == 'Filled jobs') &
            (df['Series_title_3'] == 'Actual')]

male_latest = sex_df[(sex_df['Series_title_2'] == 'Male') & 
                     (sex_df['Period'] == sex_df['Period'].max())]['Data_value'].values[0]
female_latest = sex_df[(sex_df['Series_title_2'] == 'Female') & 
                       (sex_df['Period'] == sex_df['Period'].max())]['Data_value'].values[0]

gender_data = [male_latest, female_latest]
gender_labels = ['Male', 'Female']
colors_gender = ['#3498db', '#e74c3c']

bars = ax4.bar(gender_labels, gender_data, color=colors_gender, alpha=0.8, 
               edgecolor='black', linewidth=2)

for i, val in enumerate(gender_data):
    ax4.text(i, val, f'{val:,.0f}', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

ax4.set_ylabel('Filled Jobs', fontsize=11, fontweight='bold')
ax4.set_title('Employment by Gender (Q2 2025)', fontsize=12, fontweight='bold', pad=10)
ax4.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Panel 5: Age Group Distribution
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1:])

age_df = df[(df['Group'] == 'Age by employment variable') & 
            (df['Series_title_1'] == 'Filled jobs') &
            (df['Series_title_3'] == 'Actual')]

age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', 
              '45-49', '50-54', '55-59', '60-64', '65 +']

age_latest = []
for age in age_groups:
    val = age_df[(age_df['Series_title_2'] == age) & 
                 (age_df['Period'] == age_df['Period'].max())]['Data_value'].values
    if len(val) > 0:
        age_latest.append(val[0])
    else:
        age_latest.append(0)

colors_age = sns.color_palette("viridis", len(age_groups))
bars = ax5.bar(age_groups, age_latest, color=colors_age, alpha=0.8, 
               edgecolor='black', linewidth=1.5)

for i, val in enumerate(age_latest):
    if val > 0:
        ax5.text(i, val, f'{val:,.0f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold', rotation=0)

ax5.set_xlabel('Age Group', fontsize=11, fontweight='bold')
ax5.set_ylabel('Filled Jobs', fontsize=11, fontweight='bold')
ax5.set_title('Employment Distribution by Age (Q2 2025)', fontsize=12, fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3, axis='y')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============================================================================
# Panel 6: Industry Growth Rates
# ============================================================================
ax6 = fig.add_subplot(gs[3, :])

growth_data = []
industries = ['Agriculture, Forestry and Fishing', 'Mining', 'Manufacturing', 
              'Electricity, Gas, Water and Waste Services', 'Health Care and Social Assistance',
              'Education and Training', 'Arts and Recreation Services', 'Other Services']

for industry in industries:
    ind_data = industry_df[industry_df['Series_title_2'] == industry].sort_values('Period')
    if len(ind_data) > 1:
        first_val = ind_data['Data_value'].iloc[0]
        last_val = ind_data['Data_value'].iloc[-1]
        if pd.notna(first_val) and pd.notna(last_val) and first_val > 0:
            growth_pct = ((last_val - first_val) / first_val * 100)
            growth_data.append({'Industry': industry[:30], 'Growth': growth_pct})

if len(growth_data) > 0:
    growth_df = pd.DataFrame(growth_data).sort_values('Growth', ascending=False)
    colors_growth = ['#2ecc71' if g >= 0 else '#e74c3c' for g in growth_df['Growth']]
    
    bars = ax6.bar(range(len(growth_df)), growth_df['Growth'], color=colors_growth, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax6.set_xticks(range(len(growth_df)))
    ax6.set_xticklabels(growth_df['Industry'], rotation=45, ha='right', fontsize=10)
    
    for i, val in enumerate(growth_df['Growth']):
        ax6.text(i, val, f'{val:+.1f}%', ha='center', 
                va='bottom' if val >= 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax6.set_ylabel('Growth Rate (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Employment Growth by Industry (2011-2025)', fontsize=14, fontweight='bold', pad=10)
    ax6.grid(True, alpha=0.3, axis='y')

plt.savefig('/vercel/sandbox/visualizations/00_executive_dashboard.png', bbox_inches='tight', dpi=300)
print("Executive dashboard created!")

# Also create a simple summary image
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Employment Data - Key Highlights', fontsize=20, fontweight='bold')

# Chart 1: Total employment trend
total_df_plot = total_df.copy()
dates = pd.to_datetime(total_df_plot['Period'].apply(lambda x: f"{int(x)}-{int((x % 1) * 100):02d}-01"))
ax1.plot(dates, total_df_plot['Data_value'], linewidth=3, color='#2c3e50', marker='o', markersize=5)
ax1.fill_between(dates, total_df_plot['Data_value'], alpha=0.3, color='#3498db')
ax1.set_ylabel('Total Jobs', fontsize=11, fontweight='bold')
ax1.set_title('Total Employment Growth', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Chart 2: Gender comparison
ax2.bar(gender_labels, gender_data, color=colors_gender, alpha=0.8, edgecolor='black', linewidth=2)
for i, val in enumerate(gender_data):
    ax2.text(i, val, f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_ylabel('Filled Jobs', fontsize=11, fontweight='bold')
ax2.set_title('Gender Employment (Q2 2025)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Chart 3: Top 5 age groups
top_ages_idx = np.argsort(age_latest)[-5:]
top_ages = [age_groups[i] for i in top_ages_idx]
top_values = [age_latest[i] for i in top_ages_idx]
ax3.barh(top_ages, top_values, color=sns.color_palette("mako", 5), alpha=0.8, edgecolor='black', linewidth=1.5)
for i, val in enumerate(top_values):
    ax3.text(val, i, f' {val:,.0f}', va='center', fontsize=10, fontweight='bold')
ax3.set_xlabel('Filled Jobs', fontsize=11, fontweight='bold')
ax3.set_title('Top 5 Age Groups by Employment', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Chart 4: Earnings by sector (top 4)
earnings_df = df[(df['Series_title_1'] == 'Total earnings') & 
                  (df['Series_title_3'] == 'Actual') &
                  (df['Group'] == 'Industry by employment variable')]

latest_earnings = earnings_df[earnings_df['Period'] == earnings_df['Period'].max()]
top_earnings = latest_earnings.nlargest(4, 'Data_value')

if len(top_earnings) > 0:
    colors_earn = sns.color_palette("rocket", len(top_earnings))
    bars = ax4.barh(range(len(top_earnings)), top_earnings['Data_value'], 
                    color=colors_earn, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_yticks(range(len(top_earnings)))
    ax4.set_yticklabels([ind[:30] for ind in top_earnings['Series_title_2']], fontsize=9)
    
    for i, val in enumerate(top_earnings['Data_value']):
        ax4.text(val, i, f' ${val:,.0f}M', va='center', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Total Earnings (Millions $)', fontsize=11, fontweight='bold')
    ax4.set_title('Top Industries by Earnings (Q2 2025)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

plt.savefig('/vercel/sandbox/visualizations/00_summary_highlights.png', bbox_inches='tight', dpi=300)
print("Summary highlights created!")
print("\nAll visualizations complete!")
