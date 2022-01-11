import pandas as pd
from pandas import read_csv
from datetime import datetime
import calendar
import matplotlib.pyplot as plt

df = read_csv('dataset_1/NYC_collisions_tabular.csv')

quarters = {
    'Q1' : 0,
    'Q2' : 0,
    'Q3' : 0,
    'Q4' : 0,
}

months = {
    'Jan' : 0,
    'Feb' : 0,
    'Mar' : 0,
    'Apr' : 0,
    'May' : 0,
    'Jun' : 0,
    'Jul' : 0,
    'Aug' : 0,
    'Sep' : 0,
    'Oct' : 0,
    'Nov' : 0,
    'Dec' : 0,
}

weekdays = {
    'Monday' : 0,
    'Tuesday' : 0,
    'Wednesday' : 0,
    'Thursday' : 0,
    'Friday' : 0,
    'Saturday' : 0,
    'Sunday' : 0,
}

days = {
    '1' : 0,
    '2' : 0,
    '3' : 0,
    '4' : 0,
    '5' : 0,
    '6' : 0,
    '7' : 0,
    '8' : 0,
    '9' : 0,
    '10' : 0,
    '11' : 0,
    '12' : 0,
    '13' : 0,
    '14' : 0,
    '15' : 0,
    '16' : 0,
    '17' : 0,
    '18' : 0,
    '19' : 0,
    '20' : 0,
    '21' : 0,
    '22' : 0,
    '23' : 0,
    '24' : 0,
    '25' : 0,
    '26' : 0,
    '27' : 0,
    '28' : 0,
    '29' : 0,
    '30' : 0,
    '31' : 0,
}

for day in df.CRASH_DATE:
    day = datetime.strptime(day, "%d/%m/%Y").date()
    if pd.Timestamp(day).quarter == 1:
        quarters['Q1'] += 1
    elif pd.Timestamp(day).quarter == 2:
        quarters['Q2'] += 1
    elif pd.Timestamp(day).quarter == 3:
        quarters['Q3'] += 1
    elif pd.Timestamp(day).quarter == 4:
        quarters['Q4'] += 1
    months[day.strftime('%b')] += 1
    weekdays[calendar.day_name[day.weekday()]] += 1
    days[str(day.day)] += 1

plt.title('Accidents by year quarter', pad=10.0)
plt.ylabel('Accidents', labelpad=10.0)
plt.xlabel('Quarter', labelpad=10.0)
plt.bar(quarters.keys(), quarters.values())
plt.tight_layout()
plt.savefig('images/accidents_by_year_quarter.png')
plt.close()

plt.title('Accidents by month', pad=10.0)
plt.ylabel('Accidents', labelpad=10.0)
plt.xlabel('Month', labelpad=10.0)
plt.bar(months.keys(), months.values())
plt.tight_layout()
plt.savefig('images/accidents_by_month.png')
plt.close()

plt.title('Accidents by day of week', pad=10.0)
plt.ylabel('Accidents', labelpad=10.0)
plt.xlabel('Day of week', labelpad=10.0)
plt.bar(weekdays.keys(), weekdays.values(), bottom=0.5)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('images/accidents_by_day_of_week.png')
plt.close()

plt.figure(figsize=(10,4))
plt.title('Accidents by day of month', pad=10.0)
plt.ylabel('Accidents', labelpad=10.0)
plt.xlabel('Day of month', labelpad=10.0)
plt.bar(days.keys(), days.values())
plt.tight_layout()
plt.savefig('images/accidents_by_day_of_month.png')
plt.close()

hours = [0] * 24

for hour in df.CRASH_TIME:
    hour = datetime.strptime(hour, "%H:%M").time().hour
    hours[hour] += 1

plt.figure(figsize=(10,4))
plt.title('Accidents by hour of day', pad=10.0)
plt.ylabel('Accidents', labelpad=10.0)
plt.xlabel('Hour of day', labelpad=10.0)
plt.bar(['00h', '01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h', '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h'], hours)
plt.tight_layout()
plt.savefig('images/accidents_by_hour_of_day.png')
plt.close()

age_groups = {
    '<16' : 0,
    '16-21' : 0,
    '21-25' : 0,
    '25-40' : 0,
    '40-65' : 0,
    '65<' : 0,
}

for age in df.PERSON_AGE:
    if age < 16: 
        age_groups['<16'] += 1
    elif 16 <= age and age < 21:
        age_groups['16-21'] += 1
    elif 21 <= age and age < 25: 
        age_groups['21-25'] += 1
    elif 25 <= age and age < 40: 
        age_groups['25-40'] += 1
    elif 40 <= age and age < 65:
        age_groups['40-65'] += 1
    else:
        age_groups['65<'] += 1

plt.title('Accidents by age group', pad=10.0)
plt.ylabel('Accidents', labelpad=10.0)
plt.xlabel('Age group', labelpad=10.0)
plt.bar(age_groups.keys(), age_groups.values())
plt.tight_layout()
plt.savefig('images/accidents_by_age_group.png')
plt.close()