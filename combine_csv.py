import pandas as pd

# # Read both CSV files
# df1 = pd.read_csv('updated_data_1.csv')
# df2 = pd.read_csv('updated_data_2.csv')

# # Find the maximum ID in the first file
# max_id = df1['Person ID'].max()

# # Increment all IDs in the second file by this max_id
# df2['Person ID'] = df2['Person ID'] + max_id

# # Combine the dataframes
# combined_df = pd.concat([df1, df2], ignore_index=True)

# # Save to a new CSV file
# combined_df.to_csv('general_data.csv', index=False)


# Загрузка данных
df = pd.read_csv('general_data.csv')

# 1. Числовые столбцы - считаем полную статистику
numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
               'Stress Level', 'Systolic', 'Diastolic', 'Heart Rate', 'Daily Steps']

numeric_agg = df.groupby('Occupation')[numeric_cols].agg(
    ['mean', 'std']
)

# 2. Категориальные столбцы - мода и уникальные значения
categorical_cols = ['Gender', 'BMI Category', 'Sleep Disorder']

def get_mode(x):
    mode = x.mode()
    return mode[0] if not mode.empty else None

categorical_agg = df.groupby('Occupation')[categorical_cols].agg(
    Mode=('Gender', get_mode),
    Unique_Count=('Gender', 'nunique')
    # Аналогично для других категориальных столбцов при необходимости
)

# 3. "Разворачиваем" мультииндекс для удобства чтения
numeric_agg.columns = [f'{col}_{stat}' for col, stat in numeric_agg.columns]

# 4. Объединяем результаты
result = pd.concat([numeric_agg, categorical_agg], axis=1)

# 5. Сохраняем в CSV с красивым форматированием
result.to_csv('aggregated_general_data.csv', float_format='%.2f')

print("Готово! Результат сохранён в 'aggregated_general_data.csv'")
print("\nПример результата:")
print(result.head().to_string())