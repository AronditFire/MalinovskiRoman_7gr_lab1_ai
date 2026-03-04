import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = "train.csv"

if not os.path.exists(file_path):
    print("Ошибка: файл train.csv не найден в папке проекта.")
    exit()

if os.path.getsize(file_path) == 0:
    print("Ошибка: файл train.csv пустой.")
    exit()

try:
    # Загружаем датасет
    df = pd.read_csv(file_path)
except pd.errors.EmptyDataError:
    print("Ошибка: файл train.csv пустой или повреждён.")
    exit()
except Exception as e:
    print("Ошибка при чтении файла:", e)
    exit()

print("Первые 5 строк датасета:")
print(df.head())
print()

print("Информация о датасете:")
df.info()
print()

print("Количество пропусков ДО заполнения:")
print(df.isnull().sum())
print()

# Age заполняем медианой
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

# Денежные / числовые признаки заполняем средним
numeric_fill_columns = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

for col in numeric_fill_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# Категориальные признаки заполняем модой
categorical_fill_columns = [
    "HomePlanet",
    "CryoSleep",
    "Cabin",
    "Destination",
    "VIP",
    "Name",
]

for col in categorical_fill_columns:
    if col in df.columns and not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Количество пропусков ПОСЛЕ заполнения:")
print(df.isnull().sum())
print()

# Cabin - сложный текстовый столбец, его можно удалить, чтобы не раздувать OHE
columns_to_drop = []

for col in ["PassengerId", "Name", "Cabin"]:
    if col in df.columns:
        columns_to_drop.append(col)

if columns_to_drop:
    df = df.drop(columns=columns_to_drop)

print("Столбцы после удаления лишних признаков:")
print(df.columns.tolist())
print()

numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

# Целевую переменную не нормализуем
if "Transported" in numeric_columns:
    numeric_columns.remove("Transported")

if len(numeric_columns) > 0:
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("После нормализации:")
print(df.head())
print()

categorical_columns = df.select_dtypes(include=["object", "category", "bool", "str"]).columns.tolist()

if "Transported" in categorical_columns:
    categorical_columns.remove("Transported")

if len(categorical_columns) > 0:
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

if "Transported" in df.columns:
    df["Transported"] = df["Transported"].astype(int)

print("После кодирования категориальных признаков:")
print(df.head())
print()

df.to_csv("processed_train.csv", index=False)

print("Готово! Файл processed_train.csv сохранён.")