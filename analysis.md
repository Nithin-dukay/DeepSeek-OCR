# Employment Data Analysis Report
## New Zealand Business Data Collection (2011-2025)

**Analysis Date:** November 15, 2025  
**Dataset:** employment-data.csv  
**Total Records:** 20,108  
**Time Period:** Q2 2011 - Q2 2025 (57 quarters)

---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Structure](#data-structure)
3. [Industry Analysis](#industry-analysis)
4. [Demographic Analysis](#demographic-analysis)
5. [Regional Analysis](#regional-analysis)
6. [Temporal Trends](#temporal-trends)
7. [Key Insights](#key-insights)
8. [Methodology](#methodology)

---

## Dataset Overview

### Summary Statistics
- **Total Records:** 20,108 observations
- **Time Span:** 14 years (Q2 2011 to Q2 2025)
- **Quarterly Frequency:** 57 periods
- **Data Completeness:** 91.5% (18,390 non-null values, 1,718 suppressed)
- **File Size:** 2.97 MB

### Data Dimensions
The dataset captures employment data across five major dimensions:

1. **Industry** (10 categories)
2. **Age Groups** (11 categories: 15-19, 20-24, ..., 65+)
3. **Gender** (2 categories: Male, Female)
4. **Regions** (16 regions)
5. **Territorial Authorities** (68 authorities)

### Measurement Types
- **Filled Jobs:** Number of employed positions
- **Total Earnings:** Aggregate earnings in millions of dollars
- **Adjustments:** Actual, Seasonally adjusted, and Trend values

---

## Data Structure

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| Series_reference | String | Unique identifier for each data series (370 unique codes) |
| Period | Float | Time period in YYYY.MM format (quarterly) |
| Data_value | Float | Numeric value (jobs count or earnings) |
| Suppressed | String | 'Y' if data is suppressed for confidentiality |
| STATUS | String | F (Final), R (Revised), C (Confidential) |
| UNITS | String | 'Number' (jobs) or 'Value' (earnings in millions) |
| Magnitude | Integer | 0 (actual numbers) or 6 (millions) |
| Subject | String | Always 'Business Data Collection - BDC' |
| Group | String | Dimension category (Industry/Age/Sex/Region/Territory) |
| Series_title_1 | String | Metric type (Filled jobs/Total earnings) |
| Series_title_2 | String | Category name (e.g., industry, age group) |
| Series_title_3 | String | Adjustment type (Actual/Seasonally adjusted/Trend) |
| Series_title_4 | Float | Additional descriptor (mostly null) |
| Series_title_5 | Float | Additional descriptor (mostly null) |

### Data Quality

**STATUS Code Distribution:**
- **F (Final):** 16,317 records (81.2%)
- **R (Revised):** 2,855 records (14.2%)
- **C (Confidential):** 936 records (4.7%)

**Data Suppression:**
- 1,718 records (8.5%) have suppressed values marked with 'Y'
- Suppression is used to protect confidentiality where values are too small

---

## Industry Analysis

### Industries Covered
1. Agriculture, Forestry and Fishing
2. Mining
3. Manufacturing
4. Electricity, Gas, Water and Waste Services
5. Public Administration and Safety
6. Education and Training
7. Health Care and Social Assistance
8. Arts and Recreation Services
9. Other Services
10. Total Industry (aggregate)

### Employment Levels (Seasonally Adjusted - Latest Q2 2025)

| Industry | Filled Jobs | Growth (2011-2025) |
|----------|-------------|-------------------|
| Manufacturing | 221,121 | +7.6% |
| Agriculture, Forestry and Fishing | 95,516 | +13.9% |
| Electricity, Gas, Water and Waste Services | 23,678 | +63.5% |
| Mining | 6,106 | +16.7% |

**Note:** Some industries have limited data availability in the seasonally adjusted series.

### Total Earnings Analysis (Actual Values)

| Industry | Latest Earnings (Q2 2025) | Average Earnings | Growth Rate |
|----------|---------------------------|------------------|-------------|
| Total Industry | $45,102.99M | $31,175.21M | +117.7% |
| Health Care and Social Assistance | $5,976.45M | $3,530.36M | +169.5% |
| Education and Training | $3,938.57M | $2,669.29M | +79.5% |
| Other Services | $1,177.24M | $780.34M | +121.9% |
| Arts and Recreation Services | $637.18M | $450.18M | +93.1% |

### Key Industry Insights

**Healthcare Sector:**
- Shows the strongest earnings growth at **169.5%** over the period
- Latest quarterly earnings: $5.98 billion
- Reflects aging population and increased healthcare demand

**Education Sector:**
- Earnings growth of **79.5%**
- Shows strong seasonal patterns (peaks in Q4, troughs in Q1)
- Latest quarterly earnings: $3.94 billion

**Manufacturing:**
- Relatively stable employment around 220,000 jobs
- Modest growth of 7.6% over 14 years
- Shows resilience despite global economic changes

**Agriculture:**
- Strong seasonal variation in employment
- Growth of 13.9% in seasonally adjusted terms
- Peaks in Q1 (March), troughs in Q3 (September)

---

## Demographic Analysis

### Employment by Gender (Actual Values)

**Male Employment:**
- Average: 1,015,877 jobs
- Latest (Q2 2025): 1,124,807 jobs
- Growth: **+30.6%**
- Starting (Q2 2011): 861,368 jobs

**Female Employment:**
- Average: 1,005,377 jobs
- Latest (Q2 2025): 1,134,933 jobs
- Growth: **+31.1%**
- Starting (Q2 2011): 865,507 jobs

**Gender Gap Analysis:**
- **2011:** Males had 4,139 fewer jobs than females (-0.5%)
- **2025:** Females now have 10,126 more jobs than males (+0.9%)
- The gap has **reversed and widened** in favor of female employment
- Female employment growth slightly outpaced male growth

### Employment by Age Group (Actual Values)

| Age Group | Average Jobs | Latest (Q2 2025) | Growth Rate | Key Trend |
|-----------|--------------|------------------|-------------|-----------|
| 15-19 | 114,393 | 113,927 | +15.6% | Declining in recent years |
| 20-24 | 219,864 | 214,165 | +11.6% | Slight decline from peak |
| 25-29 | 239,286 | 244,477 | +32.3% | Strong growth |
| 30-34 | 227,096 | 278,578 | **+60.8%** | Highest absolute growth |
| 35-39 | 209,145 | 270,882 | +45.1% | Strong growth |
| 40-44 | 207,265 | 239,524 | +19.4% | Moderate growth |
| 45-49 | 206,278 | 211,645 | +5.4% | Stable |
| 50-54 | 197,505 | 208,581 | +16.6% | Moderate growth |
| 55-59 | 170,064 | 184,468 | +31.5% | Growing participation |
| 60-64 | 130,227 | 157,693 | +48.8% | Strong growth |
| 65+ | 93,867 | 128,988 | **+108.5%** | Highest percentage growth |

### Age Group Insights

**Youth Employment (15-24):**
- Combined employment: ~328,000 jobs
- Growth has slowed in recent years
- Possible factors: higher education enrollment, demographic shifts

**Prime Working Age (25-54):**
- Represents the largest employment segment
- 30-34 age group shows exceptional growth (+60.8%)
- Reflects demographic bulge and career progression

**Older Workers (55+):**
- **Dramatic increase** in employment, especially 65+ (+108.5%)
- Reflects:
  - Aging population
  - Later retirement ages
  - Policy changes around retirement
  - Better health enabling longer careers
  - Economic necessity

**Most Dynamic Age Group:**
- **30-34 years:** Grew from 173,195 to 278,578 jobs (+60.8%)
- Represents peak career establishment phase

---

## Regional Analysis

### Regional Employment Distribution (Q2 2025 - Actual)

New Zealand's 16 regions show varied employment levels:

**Major Employment Centers:**
1. **Auckland:** Largest employment base (metropolitan area)
2. **Canterbury:** Second largest (includes Christchurch)
3. **Wellington:** Capital region, government employment
4. **Waikato:** Strong agricultural and service sectors

**Regional Characteristics:**
- Each region has 363 records in the dataset
- Covers all measurement types and time periods
- Shows regional economic diversity

**Territorial Authorities:**
- 68 territorial authorities tracked
- 11,488 total records (57% of dataset)
- Provides granular local employment data

---

## Temporal Trends

### Overall Employment Growth

**Total Industry Employment (Seasonally Adjusted):**
- Steady growth from 2011 to 2020
- COVID-19 impact visible in 2020 (slight disruption)
- Strong recovery post-2020
- Continued growth through 2025

### Quarterly Patterns (Seasonality)

**Average Employment by Quarter:**
- **Q1 (Jan-Mar):** Typically highest (seasonal peak in agriculture, education year-end)
- **Q2 (Apr-Jun):** Moderate levels
- **Q3 (Jul-Sep):** Typically lowest (winter season, agricultural off-season)
- **Q4 (Oct-Dec):** Rising (holiday season, retail, summer preparation)

**Seasonal Variation:**
- Most pronounced in Agriculture, Forestry and Fishing
- Education shows inverse pattern (academic calendar)
- Service industries relatively stable

### Year-over-Year Growth Patterns

**Growth Phases:**
1. **2011-2014:** Moderate growth (post-GFC recovery)
2. **2014-2019:** Steady expansion
3. **2020:** COVID-19 disruption (minimal impact in NZ)
4. **2021-2025:** Strong recovery and continued growth

**Volatility:**
- Generally low volatility in total employment
- Individual industries show more variation
- Seasonal adjustment smooths quarterly fluctuations

---

## Key Insights

### 1. Labor Market Transformation

**Aging Workforce:**
- Employment among 65+ workers has more than doubled (+108.5%)
- Reflects demographic shift and policy changes
- Challenges traditional retirement models

**Gender Parity Achievement:**
- Female employment has overtaken male employment
- Gap reversed from -0.5% to +0.9%
- Indicates successful gender equality initiatives

### 2. Sectoral Shifts

**Service Sector Dominance:**
- Healthcare earnings grew 169.5% (fastest growth)
- Education earnings grew 79.5%
- Reflects shift to service-based economy

**Traditional Sectors:**
- Manufacturing employment relatively stable
- Agriculture shows seasonal patterns but modest growth
- Mining sector small but stable

### 3. Economic Resilience

**COVID-19 Impact:**
- New Zealand showed remarkable resilience
- Employment disruption was minimal and brief
- Quick recovery to growth trajectory

**Long-term Growth:**
- Total industry earnings more than doubled (+117.7%)
- Consistent employment growth across most demographics
- Economic diversification evident

### 4. Regional Dynamics

**Geographic Distribution:**
- 16 regions with distinct economic profiles
- Urban centers (Auckland, Wellington, Canterbury) dominate
- Regional diversity in employment patterns

### 5. Data Quality and Reliability

**Comprehensive Coverage:**
- Multiple adjustment types (Actual, Seasonally adjusted, Trend)
- Allows for robust analysis
- High data completeness (91.5%)

**Suppression for Privacy:**
- 8.5% of records suppressed
- Protects confidentiality in small categories
- Maintains data integrity

---

## Methodology

### Data Processing

**Phase 1: Structure Discovery**
- Loaded complete dataset (20,108 records)
- Identified 14 columns and 5 dimensional groupings
- Analyzed data types and missing values

**Phase 2: Deep Analysis**
- Grouped data by dimensions (Industry, Age, Sex, Region)
- Calculated unique values and distributions
- Identified measurement types and units

**Phase 3: Statistical Analysis**
- Computed descriptive statistics for all series
- Calculated growth rates (2011-2025)
- Analyzed trends and patterns

**Phase 4: Visualization**
- Created 15 comprehensive visualizations
- Used multiple chart types (line, bar, heatmap, pie, correlation)
- Applied professional styling and color schemes

### Analytical Techniques

**Time Series Analysis:**
- Quarterly data from 2011 to 2025
- Seasonal adjustment comparison
- Year-over-year growth calculations

**Comparative Analysis:**
- Cross-industry comparisons
- Gender gap analysis
- Age cohort comparisons
- Regional distributions

**Statistical Methods:**
- Descriptive statistics (mean, median, std dev)
- Growth rate calculations
- Correlation analysis
- Trend identification

### Visualization Approach

**Chart Types Used:**
1. **Line Charts:** Time series trends
2. **Bar Charts:** Categorical comparisons
3. **Heatmaps:** Multi-dimensional patterns
4. **Pie Charts:** Distribution analysis
5. **Multi-panel Dashboards:** Comprehensive overviews
6. **Correlation Matrices:** Relationship analysis

**Design Principles:**
- High-resolution output (300 DPI)
- Professional color palettes
- Clear labels and titles
- Grid lines for readability
- Consistent styling across all charts

---

## Detailed Findings

### Industry Employment Trends

**Manufacturing Sector:**
- Started: ~205,000 jobs (Q2 2011)
- Current: ~221,000 jobs (Q2 2025)
- Growth: +7.6%
- Pattern: Gradual growth until 2023, slight decline 2023-2025
- Stability: Relatively low volatility

**Agriculture, Forestry and Fishing:**
- Strong seasonal patterns
- Q1 peaks (harvest season): ~100,000 jobs
- Q3 troughs (winter): ~85,000 jobs
- Seasonal variation: ~15-20%
- Overall growth: +13.9% (seasonally adjusted)

**Healthcare and Social Assistance:**
- Earnings growth: +169.5% (highest)
- Reflects:
  - Aging population increasing demand
  - Healthcare system expansion
  - Wage growth in the sector
  - Increased service provision

**Education and Training:**
- Strong seasonal patterns (academic calendar)
- Q4 peaks, Q1 troughs
- Earnings growth: +79.5%
- Stable employment base

**Utilities (Electricity, Gas, Water, Waste):**
- Smallest sector by employment
- Steady growth: +63.5%
- Capital-intensive, stable employment

**Mining:**
- Small sector (~6,000 jobs)
- Stable over time (+16.7%)
- Low volatility

### Age Demographics Deep Dive

**Youth Employment Challenges (15-24):**
- 15-19 age group peaked in 2022 (~157,000 jobs)
- Declined to ~114,000 by 2025
- Possible causes:
  - Increased tertiary education participation
  - Automation of entry-level positions
  - Demographic decline in youth population
  - Higher skill requirements

**Prime Age Workers (25-44):**
- Strongest employment growth
- 30-34 age group: +60.8% growth
- Reflects:
  - Demographic bulge (millennials)
  - Career establishment phase
  - Peak productivity years
  - Family formation and economic activity

**Mature Workers (45-64):**
- Moderate but steady growth
- 60-64 age group: +48.8% growth
- Indicates:
  - Delayed retirement
  - Experience valued in labor market
  - Financial necessity for longer careers

**Senior Workers (65+):**
- Explosive growth: +108.5%
- From ~62,000 to ~129,000 jobs
- Drivers:
  - Retirement age increases
  - Pension eligibility changes
  - Healthier aging population
  - Part-time and flexible work options
  - Economic factors

### Gender Employment Evolution

**Historical Context (Q2 2011):**
- Male: 861,368 jobs
- Female: 865,507 jobs
- Gap: -4,139 (females had 0.5% more jobs)

**Current Status (Q2 2025):**
- Male: 1,124,807 jobs
- Female: 1,134,933 jobs
- Gap: +10,126 (females have 0.9% more jobs)

**Growth Comparison:**
- Male growth: +30.6% (+263,439 jobs)
- Female growth: +31.1% (+269,426 jobs)
- Female growth outpaced male by 0.5 percentage points

**Implications:**
- Near-perfect gender parity achieved
- Slight female advantage in recent years
- Reflects:
  - Growth in female-dominated sectors (healthcare, education)
  - Improved workplace equality
  - Educational attainment parity
  - Policy effectiveness

### Regional Employment Patterns

**16 Regions Analyzed:**
- Auckland (largest urban center)
- Wellington (capital, government)
- Canterbury (Christchurch, second city)
- Waikato (agricultural heartland)
- Bay of Plenty
- Otago (Queenstown, Dunedin)
- And 10 other regions

**Regional Characteristics:**
- Urban regions: Higher employment, service-oriented
- Rural regions: Agricultural focus, seasonal patterns
- Tourism regions: Hospitality and recreation employment
- Each region: 363 data points across all periods

**Territorial Authority Granularity:**
- 68 local authorities tracked
- Provides city/district level detail
- Enables local economic planning
- 11,488 records (most detailed dimension)

---

## Temporal Trends

### Long-term Growth (2011-2025)

**Total Employment:**
- Consistent upward trajectory
- Compound annual growth rate: ~2.0%
- Resilient to economic shocks

**Economic Cycles:**
- 2011-2014: Post-GFC recovery
- 2014-2019: Expansion phase
- 2020: COVID-19 (minimal NZ impact)
- 2021-2025: Strong recovery and growth

### Seasonal Patterns

**Quarter 1 (Jan-Mar):**
- Peak for agriculture (harvest)
- Low for education (summer break)
- Generally high overall employment

**Quarter 2 (Apr-Jun):**
- Moderate levels
- Education ramp-up
- Agricultural decline

**Quarter 3 (Jul-Sep):**
- Lowest for agriculture (winter)
- Stable for most industries
- Generally lowest overall employment

**Quarter 4 (Oct-Dec):**
- Rising employment
- Retail and hospitality peak (holidays)
- Education peak (year-end)
- Agricultural preparation for summer

### Year-over-Year Growth Volatility

**Stable Periods:**
- 2012-2019: Consistent 1-3% annual growth
- Low volatility, predictable patterns

**Disruption Period:**
- 2020: COVID-19 impact minimal in NZ
- Brief slowdown, quick recovery
- Demonstrates economic resilience

**Recent Trends (2021-2025):**
- Return to growth trajectory
- Some moderation in 2024-2025
- Possible economic cooling

---

## Statistical Analysis

### Correlation Between Industries

**Highly Correlated Industries:**
- Manufacturing and Agriculture: Moderate positive correlation
- Healthcare and Education: Strong positive correlation (both service sectors)
- Most industries show positive correlation (general economic growth)

**Independent Patterns:**
- Mining shows lower correlation (commodity-driven)
- Agriculture has unique seasonal patterns

### Distribution Analysis

**Employment Distribution:**
- Right-skewed: Few large industries, many smaller sectors
- Manufacturing and Healthcare are largest employers
- Long tail of specialized industries

**Earnings Distribution:**
- Even more right-skewed than employment
- Total industry earnings: $45+ billion quarterly
- Dominated by large service sectors

### Variability Analysis

**Coefficient of Variation by Industry:**
- Agriculture: High (seasonal)
- Education: High (academic calendar)
- Healthcare: Low (stable demand)
- Manufacturing: Low (steady production)

---

## Data Suppression Patterns

### Suppression Analysis

**Total Suppressed Records:** 1,718 (8.5%)

**Reasons for Suppression:**
- Small cell sizes (confidentiality)
- Territorial authority level (granular data)
- Specific industry-region combinations
- Protects business confidentiality

**Impact on Analysis:**
- Minimal impact on aggregate trends
- Affects detailed territorial analysis
- Industry and age data largely complete

---

## Economic Insights

### Labor Market Health Indicators

**Positive Indicators:**
1. Consistent employment growth (+30-31% over 14 years)
2. Gender parity achieved and maintained
3. Older worker participation increasing (addressing aging population)
4. Service sector expansion (modern economy)
5. Regional diversity in employment

**Areas of Attention:**
1. Youth employment declining (15-24 age groups)
2. Manufacturing growth modest (potential deindustrialization)
3. Seasonal volatility in some sectors
4. Regional disparities (urban vs rural)

### Structural Changes

**Demographic Shift:**
- Workforce aging rapidly
- 65+ employment doubled
- Youth participation declining
- Implications for skills, training, retirement policy

**Sectoral Transformation:**
- Service sectors (healthcare, education) growing fastest
- Traditional sectors (manufacturing, agriculture) stable
- Knowledge economy emerging

**Gender Dynamics:**
- Female employment growth slightly faster
- Reflects sectoral shifts (growth in female-dominated sectors)
- Workplace equality improvements

---

## Recommendations for Further Analysis

### Deep Dive Opportunities

1. **Regional Economic Analysis:**
   - Compare urban vs rural employment patterns
   - Analyze regional specialization
   - Study migration patterns

2. **Wage Analysis:**
   - Calculate average earnings per job
   - Analyze wage growth by industry
   - Study gender pay gap

3. **Productivity Metrics:**
   - Earnings per employee trends
   - Industry productivity comparisons
   - Economic output analysis

4. **Forecasting:**
   - Project future employment trends
   - Model demographic impacts
   - Scenario planning

5. **Policy Impact:**
   - Analyze retirement age policy effects
   - Study gender equality initiative outcomes
   - Evaluate regional development programs

### Data Enhancement Suggestions

1. **Additional Variables:**
   - Hours worked (full-time vs part-time)
   - Skill levels and education
   - Industry sub-sectors
   - Occupation categories

2. **Higher Frequency:**
   - Monthly data for better trend detection
   - Real-time indicators

3. **Qualitative Data:**
   - Job satisfaction metrics
   - Turnover rates
   - Vacancy rates

---

## Technical Notes

### Data Quality Considerations

**Strengths:**
- High completeness (91.5%)
- Multiple adjustment types
- Long time series (14 years)
- Multiple dimensions
- Official government source

**Limitations:**
- Quarterly frequency (not monthly)
- Some suppressed values
- Limited to filled jobs (not total labor force)
- No unemployment data
- No hours worked data

### Seasonal Adjustment

**Purpose:**
- Removes regular seasonal patterns
- Reveals underlying trends
- Enables year-round comparisons

**Application:**
- Applied to most series
- Particularly important for agriculture, education
- Trend values further smooth data

### Magnitude Scaling

**Two Scales Used:**
- **Magnitude 0:** Actual numbers (jobs count)
- **Magnitude 6:** Millions (earnings values)
- Important for correct interpretation

---

## Conclusion

This comprehensive analysis of New Zealand employment data from 2011-2025 reveals a dynamic and evolving labor market characterized by:

1. **Strong Overall Growth:** Employment increased by approximately 30% across both genders
2. **Demographic Transformation:** Dramatic increase in older worker participation
3. **Gender Parity:** Achievement of near-perfect gender balance in employment
4. **Sectoral Evolution:** Shift toward service sectors, especially healthcare
5. **Economic Resilience:** Minimal COVID-19 impact, strong recovery
6. **Regional Diversity:** Varied employment patterns across 16 regions

The data demonstrates a healthy, growing economy with successful adaptation to demographic changes and economic shifts. The increasing participation of older workers and achievement of gender parity are particularly noteworthy accomplishments.

Future monitoring should focus on youth employment trends, regional disparities, and the sustainability of service sector growth.

---

## Appendix: Data Dictionary

### Series Reference Codes
- Format: BDCQ.SExxxxx
- SE: Series identifier
- A/B/C: Dimension (A=Industry, B=Sex, C=Age, etc.)
- 1/2: Metric (1=Filled jobs, 2=Earnings)
- Final letters: Adjustment type (A=Actual, S=Seasonally adjusted, T=Trend)

### Period Format
- Format: YYYY.MM
- YYYY: Year
- MM: Month (03=Q1, 06=Q2, 09=Q3, 12=Q4)

### STATUS Codes
- **F:** Final - Official published data
- **R:** Revised - Updated after initial publication
- **C:** Confidential - Suppressed for confidentiality

---

**Report End**

*This analysis was generated using Python with pandas, matplotlib, seaborn, and plotly libraries. All visualizations are included in the accompanying PDF report.*
