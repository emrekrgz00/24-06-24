import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


# Veriyi yükleyin
file_path = r'../data/UykuSaglıgı.csv'
df = pd.read_csv(file_path)
# Veri setinin ilk birkaç satırını görüntüleyelim
# print(df.head())

# Veri setinin genel bilgilerini görüntüleyelim
# print(df.info())

# Eksik değerlerin kontrolünü yapalım
# print(df.isnull().sum())

# Sütun isimlerini görüntüleyelim
# print(df.columns)

# Eksik değerleri düşürelim
df = df.dropna()

# Kategorik değişkenleri sayısal değerlere dönüştürelim
df = pd.get_dummies(df, drop_first=True)
# print(df)
for i in df:
    print(i)
# Hedef ve özellik değişkenlerini ayıralım (örneğin hedef değişken 'BMI Category' olsun)
X = df.drop('Quality of Sleep', axis=1) ##Sütunu silmek için axis=1
y = df['Quality of Sleep']

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline oluşturma
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Özellik ölçeklendirme
    ('logreg', LogisticRegression())  # Lojistik Regresyon modeli
])

# Modeli eğitelim
pipeline.fit(X_train, y_train)

# Tahmin yapalım
y_pred = pipeline.predict(X_test)

# Modelin performansını raporlayalım
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Eğitim ve test setlerindeki bazı özelliklerin dağılımını görselleştirelim
features = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']

# for feature in features:
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(X_train[feature], label='Train', fill=True)
#     sns.kdeplot(X_test[feature], label='Test', fill=True)
#     plt.title(f'Distribution of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Density')
#     plt.legend()
#     plt.savefig("../figures/"+feature+".png", dpi=320)
#     plt.show()

# # Sınıflandırma raporunu yazdır
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Confusion Matrix'i görselleştirelim
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()

# Özelliklerin önem düzeylerini görselleştirelim
feature_importance = pipeline.named_steps['logreg'].coef_[0]
features = X.columns

# Özellik önemlerini bir DataFrame'e çevirelim
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})

# Önem düzeylerine göre sıralayalım
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Görselleştirelim
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()