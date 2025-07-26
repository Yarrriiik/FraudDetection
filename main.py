import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Загрузка данных
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Удалим ненужные столбцы
drop_cols = ['profile', 'item_id', 'item_name']
X = train.drop(columns=drop_cols + ['status'])
X_test = test.drop(columns=drop_cols)
y = train['status']
profiles_test = test['profile']

# Объединяем X и X_test для единой обработки
all_data = pd.concat([X, X_test], axis=0)

# Кодируем категориальные признаки
le_dict = {}
for col in all_data.columns:
    if all_data[col].dtype == 'object':
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
        le_dict[col] = le

# Разделим обратно
X = all_data.iloc[:len(X)]
X_test = all_data.iloc[len(X):]

# Обучение и сравнение нескольких моделей
models = {
    'Random Forest': RandomForestClassifier(n_estimators=1000, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVC': SVC()
}

print("\nОценка моделей по кросс-валидации:")
best_model_name = None
best_score = 0
best_model = None

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    print(f"{name}: {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_model = model

print(f"\nЛучшая модель: {best_model_name} (точность: {best_score:.4f})")

# Обучим лучшую модель на всех данных
best_model.fit(X, y)

# Предсказание на тестовой выборке
preds = best_model.predict(X_test)

# Сохраняем submission
submission = pd.DataFrame({
    'profile': profiles_test,
    'status': preds
})

submission.to_csv('submission.csv', index=False)
print("\nФайл 'submission.csv' успешно сохранён.")

