"""
   Performance Evaluation Metrics """


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# output directory
out_dir = Path("C:/Users/Abdullah Umer/Desktop/Internee.pk Internship/Task 6/perf_reports")
out_dir.mkdir(parents=True, exist_ok=True)

# load dataset
file_path = "C:/Users/Abdullah Umer/Desktop/Internee.pk Internship/Task 6/HR-Employee-Attrition_DataSet.csv"
df = pd.read_csv(file_path)

# --- Basic cleaning & preparation ---
# Standardize column names (strip spaces)
df.columns = [c.strip() for c in df.columns]


# Helper for scaling to 0-100
def scale_0_100(series):
    # handle constant series
    if series.max() == series.min():
        return pd.Series(np.full(len(series), 50.0), index=series.index)
    return 100 * (series - series.min()) / (series.max() - series.min())

# Ensure numeric conversion for relevant columns (if any are object dtype)
numeric_cols = [
    'PerformanceRating', 'JobInvolvement', 'JobSatisfaction', 'EnvironmentSatisfaction',
    'RelationshipSatisfaction', 'TrainingTimesLastYear', 'YearsInCurrentRole',
    'YearsAtCompany', 'MonthlyIncome', 'WorkLifeBalance', 'DistanceFromHome'
]
for col in numeric_cols:
    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill NaNs sensibly (median for numeric)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Create binary OverTime flag if column exists (Yes/No)
if 'OverTime' in df.columns:
    df['OverTime_flag'] = df['OverTime'].map({'Yes':1, 'No':0})
else:
    df['OverTime_flag'] = 0

# Compute scaled components
df['perf_rating_s'] = scale_0_100(df['PerformanceRating']) if 'PerformanceRating' in df.columns else 50
df['job_involve_s'] = scale_0_100(df['JobInvolvement']) if 'JobInvolvement' in df.columns else 50
df['job_sat_s'] = scale_0_100(df['JobSatisfaction']) if 'JobSatisfaction' in df.columns else 50
df['env_sat_s'] = scale_0_100(df['EnvironmentSatisfaction']) if 'EnvironmentSatisfaction' in df.columns else 50
df['rel_sat_s'] = scale_0_100(df['RelationshipSatisfaction']) if 'RelationshipSatisfaction' in df.columns else 50

# Performance Score (weights tuned to emphasize direct performance metrics)
df['Performance_Score'] = (
    df['perf_rating_s'] * 0.45 +
    df['job_involve_s'] * 0.20 +
    df['job_sat_s'] * 0.15 +
    df['env_sat_s'] * 0.10 +
    df['rel_sat_s'] * 0.10
)

# Efficiency Score: use YearsInCurrentRole and TrainingTimesLastYear (more training -> likely more efficient)
df['years_in_role_s'] = scale_0_100(df['YearsInCurrentRole'])
df['train_last_year_s'] = scale_0_100(df['TrainingTimesLastYear'])
df['Efficiency_Score'] = (df['years_in_role_s'] * 0.6 + df['train_last_year_s'] * 0.4)

# Mentor Feedback Score: environment + relationship + work-life balance
df['worklife_s'] = scale_0_100(df['WorkLifeBalance'])
df['Mentor_Feedback_Score'] = (df['env_sat_s'] * 0.45 + df['rel_sat_s'] * 0.35 + df['worklife_s'] * 0.20)

# Engagement score: job involvement + overtime + income (income scaled)
df['monthly_income_s'] = scale_0_100(df['MonthlyIncome'])
df['Engagement_Score'] = (df['job_involve_s'] * 0.5 + df['OverTime_flag'] * 10 + df['monthly_income_s'] * 0.4)

# Overall Composite Score (final KPI) - blend of performance, efficiency, mentor feedback, engagement
df['Composite_Score'] = (
    df['Performance_Score'] * 0.5 +
    df['Efficiency_Score'] * 0.2 +
    df['Mentor_Feedback_Score'] * 0.15 +
    df['Engagement_Score'] * 0.15
)

# Round scores for readability
score_cols = ['Performance_Score','Efficiency_Score','Mentor_Feedback_Score','Engagement_Score','Composite_Score']
df[score_cols] = df[score_cols].round(2)

# Create a "Rank" based on Composite_Score
df['Composite_Rank'] = df['Composite_Score'].rank(ascending=False, method='min').astype(int)



# --- Monthly-style report generation (aggregations) ---
# Because the dataset has no explicit date column, we'll create a current-month snapshot report.
report = df.groupby(['Department']).agg(
    interns_count = ('EmployeeNumber','count'),
    avg_composite_score = ('Composite_Score','mean'),
    avg_performance = ('Performance_Score','mean'),
    avg_efficiency = ('Efficiency_Score','mean'),
    avg_mentor_feedback = ('Mentor_Feedback_Score','mean'),
    avg_engagement = ('Engagement_Score','mean')
).reset_index()

report = report.round(2)
report_file = out_dir / "monthly_performance_report.csv"
report.to_csv(report_file, index=False)

# Top performers file
top_performers = df.sort_values('Composite_Score', ascending=False).head(20)[
    ['EmployeeNumber','Age','Department','JobRole','Composite_Score','Composite_Rank','Performance_Score','Efficiency_Score']]
top_performers_file = out_dir / "top_performers.csv"
top_performers.to_csv(top_performers_file, index=False)

# Also save the full dataframe with scores
full_scored_file = out_dir / "hr_full_with_kpis.csv"
df.to_csv(full_scored_file, index=False)

# Display a concise monthly report DataFrame to the user (using pandas print)
display_report = report.copy()



# --- Visualization ---
# Use matplotlib with dark background and friendly, varied colors.
plt.style.use('dark_background')

# Helper to save fig
def save_fig(fig, filename):
    path = out_dir / filename
    fig.savefig(path, bbox_inches='tight', dpi=160)
    plt.close(fig)
    return path

# 1) Bar chart: Top 10 Composite_Score performers
fig = plt.figure(figsize=(10,6))
top10 = df.sort_values('Composite_Score', ascending=False).head(10)
bars = plt.barh(top10['EmployeeNumber'].astype(str)[::-1], top10['Composite_Score'][::-1])
plt.title("Top 10 Performers (Composite Score)", fontsize=14)
plt.xlabel("Composite Score (0-100)")
plt.gca().invert_yaxis()
# color palette
colors = ['#FFD166','#06D6A0','#118AB2','#EF476F','#06B6D4','#F9C74F','#4CC9F0','#8ECAE6','#FF7B7B','#B5838D']
for bar, c in zip(bars, colors[::-1]):
    bar.set_color(c)
save_fig(fig, "fig1_top10_bar.png")

# 2) Histogram: Composite Score distribution
fig = plt.figure(figsize=(8,5))
plt.hist(df['Composite_Score'], bins=20, edgecolor='white')
plt.title("Distribution of Composite Scores", fontsize=14)
plt.xlabel("Composite Score")
plt.ylabel("Count")
save_fig(fig, "fig2_hist_composite.png")

# 3) Boxplot: MonthlyIncome distribution by Department (top 6 depts)
fig = plt.figure(figsize=(10,6))
top_depts = df['Department'].value_counts().nlargest(6).index.tolist()
data = [df.loc[df['Department']==d,'MonthlyIncome'] for d in top_depts]
bp = plt.boxplot(data, patch_artist=True, labels=top_depts)
plt.title("Monthly Income by Department (Top 6 Departments)")
plt.ylabel("Monthly Income")
# color each box differently
box_colors = ['#264653','#2a9d8f','#e9c46a','#f4a261','#e76f51','#8ecae6']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
save_fig(fig, "fig3_box_income_dept.png")

# 4) Scatter: YearsAtCompany vs Composite Score (point size by MonthlyIncome)
fig = plt.figure(figsize=(9,6))
sizes = (df['MonthlyIncome'] - df['MonthlyIncome'].min() + 1000) / df['MonthlyIncome'].max() * 200
plt.scatter(df['YearsAtCompany'], df['Composite_Score'], s=sizes, alpha=0.85)
plt.title("Years at Company vs Composite Score (bubble size = MonthlyIncome)")
plt.xlabel("Years at Company")
plt.ylabel("Composite Score")
save_fig(fig, "fig4_scatter_years_comp.png")

# 5) Correlation heatmap (matplotlib implementation)
corr_cols = ['Composite_Score','Performance_Score','Efficiency_Score','Mentor_Feedback_Score','Engagement_Score','MonthlyIncome','YearsAtCompany']
corr = df[corr_cols].corr()
fig = plt.figure(figsize=(8,6))
plt.imshow(corr, cmap='coolwarm', aspect='equal', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha='right')
plt.yticks(range(len(corr_cols)), corr_cols)
plt.title("Correlation Matrix of KPIs & Key Features")
save_fig(fig, "fig5_corr_heatmap.png")

# 6) Bar chart: Average Composite Score by Department (sorted)
fig = plt.figure(figsize=(10,6))
dept_avg = df.groupby('Department')['Composite_Score'].mean().sort_values(ascending=False)
bars = plt.bar(dept_avg.index, dept_avg.values)
plt.title("Average Composite Score by Department")
plt.xlabel("Department")
plt.ylabel("Average Composite Score")
for i, bar in enumerate(bars):
    bar.set_color(box_colors[i % len(box_colors)])
plt.xticks(rotation=45, ha='right')
save_fig(fig, "fig6_avg_by_dept.png")

# 7) Pie chart: Attrition split (if available)
if 'Attrition' in df.columns:
    fig = plt.figure(figsize=(7,7))
    attr_counts = df['Attrition'].value_counts()
    wedges, texts, autotexts = plt.pie(attr_counts.values, labels=attr_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Attrition Distribution")
    # color wedges
    pie_colors = ['#ffb4a2','#90be6d','#00bbf0','#fcbad3']
    for w, c in zip(wedges, pie_colors):
        w.set_facecolor(c)
    save_fig(fig, "fig7_pie_attrition.png")
else:
    # create a placeholder empty plot if Attrition not present
    fig = plt.figure(figsize=(6,4))
    plt.text(0.5,0.5,"Attrition column not found", ha='center', va='center')
    plt.axis('off')
    save_fig(fig, "fig7_pie_attrition.png")

# 8) Line chart: Average Composite Score by Age groups
fig = plt.figure(figsize=(9,6))
age_bins = [18,25,30,35,40,50,60]
df['age_group'] = pd.cut(df['Age'], bins=age_bins, include_lowest=True)
age_avg = df.groupby('age_group')['Composite_Score'].mean().reset_index()
plt.plot(age_avg['age_group'].astype(str), age_avg['Composite_Score'], marker='o', linewidth=2)
plt.title("Average Composite Score by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Composite Score")
plt.xticks(rotation=45)
save_fig(fig, "fig8_line_age_groups.png")

# 9) Bar chart: Top JobRoles by Average Composite Score (top 8)
fig = plt.figure(figsize=(11,6))
role_avg = df.groupby('JobRole')['Composite_Score'].mean().sort_values(ascending=False).head(8)
bars = plt.bar(role_avg.index, role_avg.values)
plt.title("Top Job Roles by Average Composite Score (Top 8)")
plt.xlabel("Job Role")
plt.ylabel("Average Composite Score")
for i, bar in enumerate(bars):
    bar.set_color(colors[i % len(colors)])
plt.xticks(rotation=30, ha='right')
save_fig(fig, "fig9_toproles_by_score.png")

# 10) Boxplot: Composite Score distribution by OverTime flag (Yes/No)
fig = plt.figure(figsize=(8,6))
groups = []
labels = []
for val, label in [(1,'OverTime: Yes'), (0,'OverTime: No')]:
    groups.append(df.loc[df['OverTime_flag']==val, 'Composite_Score'])
    labels.append(label)
bp = plt.boxplot(groups, labels=labels, patch_artist=True)
plt.title("Composite Score by OverTime Status")
for patch, color in zip(bp['boxes'], ['#06D6A0','#118AB2']):
    patch.set_facecolor(color)
save_fig(fig, "fig10_box_overtime.png")

# --- Save a brief textual summary to accompany the report ---
summary_txt = out_dir / "summary.txt"
with open(summary_txt, "w") as f:
    f.write("Monthly Performance Report Summary\n")
    f.write("===============================\n\n")
    f.write(f"Total records analyzed: {len(df)}\n")
    f.write(f"Departments included: {', '.join(df['Department'].unique())}\n\n")
    f.write("Key KPIs created: Performance_Score, Efficiency_Score, Mentor_Feedback_Score, Engagement_Score, Composite_Score\n\n")
    f.write("Files generated:\n")
    for p in [report_file, top_performers_file, full_scored_file, summary_txt] + list(out_dir.glob("fig*.png")):
        f.write(f"- {p.name}\n")



# Use pandas display for small outputs; we'll print head of top performers and report.
print("Monthly report saved to:", report_file)
print("Top performers saved to:", top_performers_file)
print("All scored HR data saved to:", full_scored_file)
print("All figures saved to the folder:", out_dir)
print("\nTop performers (top 10):\n")
print(top_performers.head(10).to_string(index=False))

# Display the report DataFrame (first rows)
print("\nDepartment summary (monthly-style report):\n")
print(display_report.to_string(index=False))

# Provide file paths for download in assistant response
generated_files = {
    "monthly_report": str(report_file),
    "top_performers": str(top_performers_file),
    "full_scored_data": str(full_scored_file),
    "summary_text": str(summary_txt),
    "figures_folder": str(out_dir)
}

generated_files







