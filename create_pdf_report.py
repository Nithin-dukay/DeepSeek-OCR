from fpdf import FPDF
import os
from datetime import datetime

class EmploymentReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Employment Data Analysis Report', new_x="LMARGIN", new_y="NEXT", align='C')
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 5, 'New Zealand Business Data Collection (2011-2025)', new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

# Create PDF
print("Creating PDF report...")
pdf = EmploymentReportPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Cover Page
pdf.add_page()
pdf.ln(40)
pdf.set_font('Helvetica', 'B', 28)
pdf.cell(0, 15, 'Employment Data Analysis', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.set_font('Helvetica', 'B', 20)
pdf.cell(0, 12, 'New Zealand', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.ln(10)
pdf.set_font('Helvetica', '', 14)
pdf.cell(0, 8, 'Comprehensive Visual Analysis Report', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.ln(5)
pdf.set_font('Helvetica', 'I', 12)
pdf.cell(0, 8, 'Period: Q2 2011 - Q2 2025', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.ln(20)
pdf.set_font('Helvetica', '', 11)
pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%B %d, %Y")}', new_x="LMARGIN", new_y="NEXT", align='C')

# Executive Summary
pdf.add_page()
pdf.set_font('Helvetica', 'B', 14)
pdf.set_fill_color(52, 152, 219)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 10, 'Executive Summary', new_x="LMARGIN", new_y="NEXT", fill=True)
pdf.set_text_color(0, 0, 0)
pdf.ln(3)

pdf.set_font('Helvetica', '', 10)
pdf.multi_cell(0, 6, 'This report analyzes 20,108 employment records from New Zealand covering Q2 2011 to Q2 2025.')
pdf.ln(3)

# Add visualizations
visualizations = [
    ('01_industry_trends.png', '1. Employment Trends by Industry', 
     'Evolution of filled jobs across major industries (seasonally adjusted).'),
    ('02_total_employment_comparison.png', '2. Total Industry Employment', 
     'Comparison of actual, seasonally adjusted, and trend values.'),
    ('03_age_group_trends.png', '3. Employment by Age Group', 
     'Employment patterns across age demographics.'),
    ('04_gender_employment.png', '4. Employment by Gender', 
     'Male and female employment trends showing convergence.'),
    ('05_industry_earnings.png', '5. Total Earnings by Industry', 
     'Earnings trends for key service industries.'),
    ('06_latest_industry_employment.png', '6. Current Employment by Industry', 
     'Snapshot of Q2 2025 employment levels.'),
    ('07_age_heatmap.png', '7. Age Group Employment Heatmap', 
     'Employment intensity across age groups over time.'),
    ('08_regional_employment.png', '8. Employment by Region', 
     'Regional distribution across New Zealand.'),
    ('09_seasonal_patterns.png', '9. Seasonal Employment Patterns', 
     'Average employment by quarter.'),
    ('10_gender_analysis.png', '10. Gender Employment Analysis', 
     'Detailed gender trends and employment gap.'),
    ('11_industry_growth.png', '11. Industry Growth Rates', 
     'Percentage growth by industry (2011-2025).'),
    ('12_age_distribution.png', '12. Age Distribution (Q2 2025)', 
     'Current employment proportion by age.'),
    ('13_industry_dashboard.png', '13. Industry Overview Dashboard', 
     'Multi-panel view of key metrics.'),
    ('14_yoy_growth.png', '14. Year-over-Year Growth', 
     'Annual growth rates and economic cycles.'),
    ('15_correlation_matrix.png', '15. Industry Correlation Matrix', 
     'Employment relationships between industries.'),
]

for img_file, title, description in visualizations:
    img_path = f'/vercel/sandbox/visualizations/{img_file}'
    if os.path.exists(img_path):
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_fill_color(52, 152, 219)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, description)
        pdf.ln(3)
        pdf.image(img_path, x=10, w=190)

# Save PDF
output_path = '/vercel/sandbox/employment_analysis_report.pdf'
pdf.output(output_path)
print(f"\nPDF Report created successfully!")
print(f"Location: {output_path}")
print(f"Total pages: {pdf.page_no()}")
