import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Загрузка данных
data = pd.read_csv('general_data.csv')  # Замените на ваш файл

# Удаляем Person ID, так как он не нужен для кластеризации
data = data.drop('Person ID', axis=1)

# Определяем типы признаков
numeric_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 
                   'Physical Activity Level', 'Stress Level', 
                   'Systolic', 'Diastolic', 'Heart Rate', 'Daily Steps']

categorical_features = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']

# Создаем преобразователь для обработки данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Применяем преобразования
processed_data = preprocessor.fit_transform(data)

# Кластеризация (возьмем 4 кластера для примера)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(processed_data)

# Добавляем метки кластеров в исходные данные
data['Cluster'] = clusters

# Создаем DataFrame с центроидами
# Для числовых признаков - среднее по кластеру
numeric_centroids = data.groupby('Cluster')[numeric_features].mean()

# Для категориальных - мода (наиболее частое значение)
categorical_centroids = data.groupby('Cluster')[categorical_features].agg(lambda x: x.mode()[0])

# Объединяем центроиды
centroids = pd.concat([numeric_centroids, categorical_centroids], axis=1)

# Заменяем исходные данные на центроиды
clustered_data = data.copy()
for feature in numeric_features + categorical_features:
    clustered_data[feature] = clustered_data['Cluster'].map(centroids[feature])

# Удаляем столбец с метками кластеров (если не нужен)
clustered_data = clustered_data.drop('Cluster', axis=1)

# Сохраняем результат
clustered_data.to_csv('clustered_general_data.csv', index=False)

# Дополнительно: сохраняем информацию о центроидах
centroids.to_csv('cluster_general_data_info.csv')

print("Кластеризация завершена! Результаты сохранены в файлы:")
print("- clustered_general_data.csv - данные, замененные на центроиды")
print("- cluster_general_data_info - информация о центроидах кластеров")