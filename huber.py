# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import HuberRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from sklearn.utils import resample
# import scipy.stats as stats

# # Загрузка данных
# data = pd.read_csv('updated_data_2.csv')  # Замените на имя вашего файла

# # Выбор признаков и целевой переменной
# features = ['Occupation', 'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level']
# target = 'Stress Level'
# X = data[features]
# y = data[target]

# # Преобразование категориальных признаков - кодируем Occupation как единый признак
# occupation_encoder = LabelEncoder()
# X['Occupation'] = occupation_encoder.fit_transform(X['Occupation'])

# # Разделение данных
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Создание и обучение модели Хьюбера
# huber = HuberRegressor(epsilon=1.35, max_iter=1000)
# huber.fit(X_train, y_train)

# # Предсказание и оценка модели
# y_pred = huber.predict(X_test)
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)

# # Функция для расчета доверительных интервалов с помощью бутстрепа
# def bootstrap_confidence_intervals(y_true, y_pred, metric, n_bootstraps=1000, ci=95):
#     bootstrapped_scores = []
#     for _ in range(n_bootstraps):
#         indices = resample(np.arange(len(y_true)))
#         if metric == r2_score:
#             score = metric(y_true.iloc[indices], y_pred[indices])
#         else:
#             score = metric(y_true.iloc[indices], y_pred[indices])
#         bootstrapped_scores.append(score)
    
#     alpha = (100 - ci) / 2
#     lower = np.percentile(bootstrapped_scores, alpha)
#     upper = np.percentile(bootstrapped_scores, 100 - alpha)
#     return lower, upper

# # Расчет доверительных интервалов для метрик
# r2_ci = bootstrap_confidence_intervals(y_test, y_pred, r2_score)
# mae_ci = bootstrap_confidence_intervals(y_test, y_pred, mean_absolute_error)
# mse_ci = bootstrap_confidence_intervals(y_test, y_pred, mean_squared_error)

# # Анализ важности признаков
# feature_importance = pd.DataFrame({
#     'Feature': features,
#     'Importance': np.abs(huber.coef_)
# }).sort_values('Importance', ascending=False)

# # График важности признаков (вертикальный)
# plt.figure(figsize=(10, 6))
# bars = plt.bar(feature_importance['Feature'], feature_importance['Importance'], color='#1f77b4')
# plt.xlabel('Признак', fontsize=12)
# plt.ylabel('Абсолютная важность признака', fontsize=12)
# plt.title('Важность признаков для уровня стресса\n(Робастная регрессия Хьюбера)', fontsize=14)

# # Добавление значений важности на график
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,  
#              f'{height:.3f}', ha='left', va='center', fontsize=10)

# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Подготовка данных для графика метрик
# metrics = ['R²', 'MAE', 'MSE']
# values = [r2, mae, mse]
# errors = [
#     [values[0] - r2_ci[0], r2_ci[1] - values[0]],
#     [values[1] - mae_ci[0], mae_ci[1] - values[1]],
#     [values[2] - mse_ci[0], mse_ci[1] - values[2]]
# ]

# # График метрик с доверительными интервалами
# plt.figure(figsize=(10, 6))
# bars = plt.bar(metrics, values, color=['#4CAF50', '#FFC107', '#F44336'], 
#                yerr=np.array(errors).T, capsize=10, alpha=0.8)

# plt.ylabel('Значение', fontsize=12)
# plt.title('Оценка качества модели с доверительными интервалами (95%)', fontsize=14)

# # Добавление значений на столбцы
# for bar, value, error in zip(bars, values, errors):
#     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
#              f'{value:.3f}\n±{np.mean(error):.3f}', 
#              ha='center', va='bottom', fontsize=10)

# # Линия для R² = 0
# plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Вывод результатов
# print(f"\nОценка модели:")
# print(f"R² (коэффициент детерминации): {r2:.3f} [95% ДИ: {r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")
# print(f"MAE (средняя абсолютная ошибка): {mae:.3f} [95% ДИ: {mae_ci[0]:.3f}, {mae_ci[1]:.3f}]")
# print(f"MSE (среднеквадратичная ошибка): {mse:.3f} [95% ДИ: {mse_ci[0]:.3f}, {mse_ci[1]:.3f}]")

# print("\nВажность признаков:")
# print(feature_importance.sort_values('Importance', ascending=False))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils import resample

# Загрузка данных
data = pd.read_csv('updated_data_1.csv')  # Замените на имя вашего файла

# Выбор признаков и целевой переменной
features = ['Occupation', 'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level']
target = 'Stress Level'
X = data[features]
y = data[target]

# Преобразование категориальных признаков
le = LabelEncoder()
X['Occupation'] = le.fit_transform(X['Occupation'])

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Параметры для поиска оптимального epsilon
param_grid = {'epsilon': np.arange(1.0, 3.5, 0.1)}

# Поиск оптимального epsilon с помощью GridSearchCV
huber_tuned = GridSearchCV(
    HuberRegressor(max_iter=1000),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
huber_tuned.fit(X_train, y_train)

# Лучшие параметры и модель
best_epsilon = huber_tuned.best_params_['epsilon']
best_model = huber_tuned.best_estimator_

print(f"Оптимальный параметр epsilon: {best_epsilon:.2f}")
print(f"Лучший R² при кросс-валидации: {huber_tuned.best_score_:.3f}")

# Предсказание и оценка модели
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Функция для расчета доверительных интервалов с помощью бутстрепа
def bootstrap_confidence_intervals(y_true, y_pred, metric, n_bootstraps=1000, ci=95):
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)))
        if metric == r2_score:
            score = metric(y_true.iloc[indices], y_pred[indices])
        else:
            score = metric(y_true.iloc[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrapped_scores, alpha)
    upper = np.percentile(bootstrapped_scores, 100 - alpha)
    return lower, upper

# Расчет доверительных интервалов для метрик
r2_ci = bootstrap_confidence_intervals(y_test, y_pred, r2_score)
mae_ci = bootstrap_confidence_intervals(y_test, y_pred, mean_absolute_error)
mse_ci = bootstrap_confidence_intervals(y_test, y_pred, mean_squared_error)

# Анализ важности признаков
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': np.abs(best_model.coef_)
}).sort_values('Importance', ascending=False)  # Сортировка по возрастанию важности

# График важности признаков (вертикальный)
plt.figure(figsize=(10, 6))
bars = plt.bar(feature_importance['Feature'], feature_importance['Importance'], color='#1f77b4')
plt.xlabel('Признак', fontsize=12)
plt.ylabel('Абсолютная важность признака', fontsize=12)
plt.title(f'Важность признаков для уровня стресса\n(Робастная регрессия Хьюбера, ε={best_epsilon:.2f})', 
          fontsize=16, pad=20)

# Добавление значений важности на график
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,  
             f'{height:.3f}', ha='left', va='center', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
# # График важности признаков (ВЕРТИКАЛЬНЫЙ с горизонтальными признаками)
# plt.figure(figsize=(12, 8))

# # Создаем горизонтальные бары (barh) с сортировкой по возрастанию важности
# bars = plt.barh(feature_importance['Feature'], 
#                 feature_importance['Importance'], 
#                 color=plt.cm.viridis(np.linspace(0, 0.8, len(features))[::-1]))

# plt.xlabel('Абсолютное влияние на уровень стресса', fontsize=14, labelpad=15)
# plt.title(f'Важность признаков для уровня стресса\n(Робастная регрессия Хьюбера, ε={best_epsilon:.2f})', 
#           fontsize=16, pad=20)

# # Добавление значений важности на график
# for bar in bars:
#     width = bar.get_width()
#     plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
#              f'{width:.3f}', ha='left', va='center', fontsize=12)

# # Вертикальная линия для лучшей визуализации
# plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
# plt.show()

# Подготовка данных для графика метрик
metrics = ['R²', 'MAE', 'MSE']
values = [r2, mae, mse]
errors = [
    [values[0] - r2_ci[0], r2_ci[1] - values[0]],
    [values[1] - mae_ci[0], mae_ci[1] - values[1]],
    [values[2] - mse_ci[0], mse_ci[1] - values[2]]
]

# График метрик с доверительными интервалами
plt.figure(figsize=(12, 8))
bars = plt.bar(metrics, values, color=['#4CAF50', '#FFC107', '#F44336'], 
               yerr=np.array(errors).T, capsize=15, alpha=0.85, width=0.6)

plt.ylabel('Значение', fontsize=14, labelpad=15)
plt.title('Оценка качества модели с доверительными интервалами (95%)', fontsize=16, pad=20)

# Добавление значений на столбцы
for bar, value, error in zip(bars, values, errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}\n±{np.mean(error):.3f}', 
             ha='center', va='bottom', fontsize=12)

# Линия для R² = 0
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Установка пределов для лучшего отображения
min_y = min(min(values) * 1.1, -0.1) if min(values) < 0 else 0
max_y = max(values) * 1.3
plt.ylim(min_y, max_y)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Вывод результатов
print(f"\nОценка модели:")
print(f"R² (коэффициент детерминации): {r2:.3f} [95% ДИ: {r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")
print(f"MAE (средняя абсолютная ошибка): {mae:.3f} [95% ДИ: {mae_ci[0]:.3f}, {mae_ci[1]:.3f}]")
print(f"MSE (среднеквадратичная ошибка): {mse:.3f} [95% ДИ: {mse_ci[0]:.3f}, {mse_ci[1]:.3f}]")

print("\nВажность признаков (по убыванию):")
print(feature_importance.sort_values('Importance', ascending=False))

# Дополнительный график: Зависимость R² от epsilon
plt.figure(figsize=(10, 6))
results = pd.DataFrame(huber_tuned.cv_results_)
plt.plot(results['param_epsilon'], results['mean_test_score'], marker='o', linestyle='-', color='b')
plt.fill_between(results['param_epsilon'],
                 results['mean_test_score'] - results['std_test_score'],
                 results['mean_test_score'] + results['std_test_score'],
                 alpha=0.2, color='b')
plt.axvline(x=best_epsilon, color='r', linestyle='--', label=f'Лучший ε = {best_epsilon:.2f}')
plt.xlabel('Значение epsilon (c)', fontsize=12)
plt.ylabel('R² (кросс-валидация)', fontsize=12)
plt.title('Зависимость качества модели от параметра epsilon', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('epsilon_tuning.png', dpi=300)
plt.show()