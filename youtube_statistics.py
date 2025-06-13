# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
data = pd.read_csv('Global YouTube Statistics.csv', encoding='unicode_escape')

# Preprocessing
data = data.drop_duplicates()
print("Data Headers and missing values ")
print(data.isnull().sum(), '\n')
print("Total number of missing values in the data")
print(data.isnull().sum().sum(), '\n')
data = data.fillna(0)
print("Total number of missing values in the data ,After filling the '0'")
print(data.isnull().sum().sum(), '\n')

# 1. Top 10 YouTube channels based on subscribers
top_10_channels = data.sort_values('subscribers', ascending=False).head(10)
print("Top 10 Youtube Channels by Subscribers:")
print(top_10_channels[['Youtuber', 'subscribers']])
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Youtuber', y='subscribers', data=top_10_channels, color='maroon')
plt.xticks(rotation=90)
for bar in ax.patches:
    value = int(bar.get_height())
    ax.text(bar.get_x() + bar.get_width()/2, value, f'{value:,.0f}', ha='center', va='bottom')
plt.xlabel('YouTube Channels')
plt.ylabel('Number of Subscribers')
plt.title('Top 10 YouTube Channels by Subscribers')
plt.tight_layout()
plt.show()

# 2. Highest average subscribers by category
avg_subs_by_category = data.groupby('category')['subscribers'].mean().sort_values(ascending=False)
highest_avg_subs_category = avg_subs_by_category.index[0]
print(f"Category with highest average subscribers: {highest_avg_subs_category}")
categories_list = data.groupby(['category']).size().reset_index(name='count')
print(categories_list, "\n")
plt.figure(figsize=(12, 8))
ax1 = sns.barplot(x='category', y='count', data=categories_list, palette="Set2")
plt.xticks(rotation=90)
for bar in ax1.patches:
    value = int(bar.get_height())
    ax1.text(bar.get_x() + bar.get_width()/2, value, f'{value:,.0f}', ha='center', va='bottom')
plt.xlabel('YouTube Categories')
plt.ylabel('Number of Channels')
plt.title('YouTube Channels Categories')
plt.tight_layout()
plt.show()

# Channel types
channel_list = data.groupby(['channel_type']).size().reset_index(name='count')
print(channel_list, "\n")
plt.figure(figsize=(12, 8))
ax2 = sns.barplot(x='channel_type', y='count', data=channel_list, palette="Set2")
plt.xticks(rotation=90)
for bar in ax2.patches:
    value = int(bar.get_height())
    ax2.text(bar.get_x() + bar.get_width()/2, value, f'{value:,.0f}', ha='center', va='bottom')
plt.xlabel('YouTube Channels')
plt.ylabel('Number of Channels')
plt.title('YouTube Channel Types')
plt.tight_layout()
plt.show()

# 3. Average uploads per category
avg_uploads_by_category = data.groupby('category')['uploads'].mean().reset_index(name="avg_upload")
print("Average uploads by category:")
print(avg_uploads_by_category)
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='category', y='avg_upload', data=avg_uploads_by_category, palette="husl")
plt.xticks(rotation=90)
for bar in ax.patches:
    value = int(bar.get_height())
    ax.text(bar.get_x() + bar.get_width()/2, value, f'{value:,.0f}', ha='center', va='bottom')
plt.xlabel('YouTube Categories')
plt.ylabel('Average Uploads')
plt.title('Average Number of Uploads by Category')
plt.tight_layout()
plt.show()

# 4. Top 5 countries by number of YouTube channels
top_5_countries = data['Country'].value_counts().head(5).reset_index(name='top_5_country')
top_countries = data['Country'].value_counts().reset_index(name='top_country')
print("Countries by Number of YouTube Channels:")
print(top_countries)
print("Top 5 Countries:")
print(top_5_countries)
plt.figure(figsize=(12, 8))
sns.barplot(x='top_country', y='Country', data=top_countries, palette="husl")
plt.xlabel('Number of Channels')
plt.ylabel('Countries')
plt.title('All Countries by YouTube Channel Count')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 8))
sns.barplot(x='top_5_country', y='Country', data=top_5_countries, palette="husl")
plt.xlabel('Number of Channels')
plt.ylabel('Countries')
plt.title('Top 5 Countries by YouTube Channel Count')
plt.tight_layout()
plt.show()

# 5. Distribution of channel types across categories
channel_type_dist = data.groupby(['category', 'channel_type']).size().reset_index(name='group_count')
pivot_table = channel_type_dist.pivot(index='category', columns='channel_type', values='group_count').fillna(0)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title('Channel Type Distribution Across Categories')
plt.tight_layout()
plt.show()

# 6. Correlation between Subscribers and Video Views
top_10_youtubers = data.sort_values(by='subscribers', ascending=False).head(10)
print(top_10_youtubers[['Youtuber', 'subscribers', 'video views']])

fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar plot for video views
bar_width = 0.4
index = range(len(top_10_youtubers['Youtuber']))
bars = ax1.bar(index, top_10_youtubers['video views'], bar_width, color='b', label='Video Views')

# Line plot for subscribers on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(index, top_10_youtubers['subscribers'], color='r', marker='o', linestyle='-', linewidth=2, label='Subscribers')

# X-axis labels
ax1.set_xticks(index)
ax1.set_xticklabels(top_10_youtubers['Youtuber'], rotation=90)

ax1.set_xlabel('YouTuber')
ax1.set_ylabel('Video Views', color='b')
ax2.set_ylabel('Subscribers', color='r')
plt.title('Subscribers and Video Views of Top 10 YouTubers')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.subplots_adjust()
plt.show()

# 7. Average Monthly Earnings by Category
earnings_by_category_low = data.groupby('category')[['lowest_monthly_earnings']].mean()
earnings_by_category_high = data.groupby('category')[['highest_monthly_earnings']].mean()

print("Monthly Earnings by Category (Lowest):")
print(earnings_by_category_low)
print("\nMonthly Earnings by Category (Highest):")
print(earnings_by_category_high)

# Combined bar plot
earnings_by_category = data.groupby('category')[['lowest_monthly_earnings', 'highest_monthly_earnings']].mean().reset_index()

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='category', y='lowest_monthly_earnings', data=earnings_by_category, color='blue', label='Lowest Monthly Earnings')
ax = sns.barplot(x='category', y='highest_monthly_earnings', data=earnings_by_category, color='red', label='Highest Monthly Earnings', alpha=0.7)

plt.xticks(rotation=90)
plt.xlabel('YouTube Categories')
plt.ylabel('Average Monthly Earnings')
plt.title('Average Monthly Earnings for Different Categories')

# Annotate bars
bars = ax.patches
for i, bar in enumerate(bars):
    value = int(bar.get_height())
    color = 'blue' if i < len(bars) // 2 else 'red'
    ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:,.0f}', ha='center', va='bottom', color=color)

plt.legend()
plt.tight_layout()
plt.show()

# 8. Subscribers Gained in Last 30 Days
subscribers_gained_last_30_days = data['subscribers_for_last_30_days'].sum()

subs_gained = data.groupby(['channel_type', 'subscribers_for_last_30_days']).size().reset_index(name='sub_gained')
print("Subscribers gained in the last 30 days across all channels:")
print(subs_gained)

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='channel_type', y='subscribers_for_last_30_days', data=subs_gained, palette="husl")
plt.xticks(rotation=90)

for bar in ax.patches:
    value = int(bar.get_height())
    ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:,.0f}', ha='center', va='bottom')

plt.xlabel('YouTube Channel Types')
plt.ylabel('Subscribers Gained in Last 30 Days')
plt.title('Subscribers Gained in the Last 30 Days by Channel Type')
plt.tight_layout()
plt.show()

# 9. Detect Outliers in Yearly Earnings
Q1 = data['highest_yearly_earnings'].quantile(0.25)
Q3 = data['highest_yearly_earnings'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['highest_yearly_earnings'] < lower_bound) | (data['highest_yearly_earnings'] > upper_bound)]
data['index'] = data.index

plt.figure(figsize=(10, 6))
sns.scatterplot(x='index', y='highest_yearly_earnings', data=data, color='blue', alpha=0.6, label='Data Points')
sns.scatterplot(x='index', y='highest_yearly_earnings', data=outliers, color='red', s=100, label='Outliers')

plt.xlabel('Index')
plt.ylabel('Highest Yearly Earnings (USD)')
plt.title('Scatter Plot of Highest Yearly Earnings with Outliers Highlighted')
plt.legend()
plt.tight_layout()
plt.show()

# Print outlier channel names
outlier_channel_names = outliers['Youtuber'].tolist()
print("Outlier YouTube Channel Names:")
print(pd.DataFrame(outlier_channel_names))


# 10. Distribution of Channel Creation Dates
yearly_channel_count = data['created_year'].value_counts().reset_index()
yearly_channel_count.columns = ['created_year', 'channel_count']

all_years = pd.DataFrame({'created_year': range(2000, yearly_channel_count['created_year'].max() + 1)})
yearly_channel_count = pd.merge(all_years, yearly_channel_count, on='created_year', how='left')
yearly_channel_count['channel_count'] = yearly_channel_count['channel_count'].fillna(0)
yearly_channel_count = yearly_channel_count.sort_values(by='created_year')

plt.figure(figsize=(10, 6))
sns.lineplot(x='created_year', y='channel_count', data=yearly_channel_count, marker='o', color='blue')
plt.title('Number of YouTube Channels Created by Year')
plt.xlabel('Year')
plt.ylabel('Number of Channels')
plt.xlim(2000, yearly_channel_count['created_year'].max())
plt.xticks(yearly_channel_count['created_year'], rotation=45)
plt.tight_layout()
plt.show()

# 11. Gross Tertiary Education Enrollment vs Channel Type
gross = data.groupby(['channel_type', 'Gross tertiary education enrollment (%)']).size().reset_index(name='average_gross_of Channels')
print(gross)

pivot_table = gross.pivot(index='Gross tertiary education enrollment (%)', columns='channel_type', values='average_gross_of Channels')
pivot_table = pivot_table.fillna(0)

# Stacked Bar Plot
pivot_table.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab20')
plt.title('Distribution of Channel Types across Gross Tertiary Education Enrollment')
plt.xlabel('Gross Tertiary Education Enrollment (%)')
plt.ylabel('Number of Channels')
plt.legend(title='Channel Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# 12. Unemployment Rate in Top 10 Countries by YouTube Channel Count
top_10_countries = data['Country'].value_counts().head(10).index
print("Top 10 countries:")
print(top_10_countries)

unemployment_rates = data[data['Country'].isin(top_10_countries)][['Country', 'Unemployment rate']].dropna()
print("Unemployment rates:")
print(unemployment_rates)

plt.figure(figsize=(12, 6))
sns.barplot(x='Country', y='Unemployment rate', data=unemployment_rates, palette='viridis')
plt.xlabel('Country')
plt.ylabel('Unemployment Rate (%)')
plt.title('Unemployment Rate by Country for Top 10 Countries with Most YouTube Channels')
plt.tight_layout()
plt.show()
# 13. Urban Population Percentage Analysis
relevant_data = data.drop_duplicates(subset=['Country'])
relevant_data['Urban_population'] = pd.to_numeric(relevant_data['Urban_population'], errors='coerce')
relevant_data['Population'] = pd.to_numeric(relevant_data['Population'], errors='coerce')

relevant_data = relevant_data[(relevant_data['Urban_population'] != 0) & (relevant_data['Population'] != 0)]

relevant_data['Urban population percentage'] = (relevant_data['Urban_population'] / relevant_data['Population']) * 100
relevant_data['Average Urban Population'] = relevant_data.apply(lambda row: row['Urban_population'] / row['Population'], axis=1)

average_urban_population_percentage = relevant_data['Urban population percentage'].mean()

plt.figure(figsize=(14, 10))

# Plot 1: Urban Population Percentage
plt.subplot(2, 1, 1)
ax1 = sns.barplot(x='Country', y='Urban population percentage', data=relevant_data, palette='viridis')
plt.axhline(average_urban_population_percentage, color='red', linestyle='--', label=f'Average: {average_urban_population_percentage:.2f}%')
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.xlabel('Country')
plt.ylabel('Urban Population Percentage (%)')
plt.title('Urban Population Percentage by Country')
plt.legend()
plt.xticks(rotation=90)

# Plot 2: Average Urban Population
plt.subplot(2, 1, 2)
sns.lineplot(x='Country', y='Average Urban Population', data=relevant_data, marker='o', color='b')
plt.xlabel('Country')
plt.ylabel('Average Urban Population')
plt.title('Average Urban Population by Country')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

print(f"The average urban population percentage in countries with YouTube channels is: {average_urban_population_percentage:.2f}%")

# 14. Channel Distribution by Geographic Coordinates
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Longitude', y='Latitude', hue='channel_type', data=data, s=100, palette='viridis', legend='full')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Distribution of YouTube Channels by Latitude and Longitude')
plt.legend(loc='lower left', title='Channel Type')
plt.tight_layout()
plt.show()

# 15. Subscribers vs Population by Country
data['Country'] = data['Country'].astype(str)
data['Country'] = pd.Categorical(data['Country'], categories=data['Country'].unique(), ordered=True)

plt.figure(figsize=(10, 6))
sns.lineplot(x='Country', y='subscribers', data=data, marker='o', color='b', label='Subscribers')
sns.lineplot(x='Country', y='Population', data=data, marker='o', color='r', label='Population')

plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Subscribers and Population by Country')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# 16. Comparison of Channel Types for Top 10 Countries
top_10_countries = data['Country'].value_counts().head(10).index
top_10_data = data[data['Country'].isin(top_10_countries)]

plt.figure(figsize=(12, 8))
sns.countplot(y='Country', hue='channel_type', data=top_10_data, palette='viridis')
plt.xlabel('Count')
plt.ylabel('Country')
plt.title('Comparison of Channel Types for Top 10 Countries')
plt.legend(title='Channel Type')
plt.tight_layout()
plt.grid()
plt.show()

# 17. Average Subscribers Gained in Last 30 Days by Channel Type
average_subs = data.groupby('channel_type')['subscribers_for_last_30_days'].mean().reset_index()

plt.figure(figsize=(10, 6))
bars = sns.barplot(x='channel_type', y='subscribers_for_last_30_days', data=average_subs, palette='viridis')
plt.title('Average Subscribers Gained Last 30 Days by Channel Type')
plt.xlabel('Channel Type')
plt.ylabel('Average Subscribers Gained Last 30 Days')

# Annotate bars with values
for bar in bars.patches:
    bars.annotate(format(bar.get_height(), '.1f'), 
                  (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                  ha='center', va='center', size=12, xytext=(0, 8), 
                  textcoords='offset points')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 18. Average Video Views for Last 30 Days by Channel Type
average_views = data.groupby('channel_type')['video_views_for_the_last_30_days'].mean().reset_index()

plt.figure(figsize=(10, 6))
bars = sns.barplot(x='channel_type', y='video_views_for_the_last_30_days', data=average_views, palette='husl')
plt.title('Average Video Views for the Last 30 Days by Channel Type')
plt.xlabel('Channel Type')
plt.ylabel('Average Video Views for the Last 30 Days')

# Annotate bars vertically
for bar in bars.patches:
    bars.annotate(format(bar.get_height(), '.1f'),
                  (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                  ha='center', va='bottom', size=12, xytext=(0, 10),
                  textcoords='offset points', rotation=90)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
