import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# === Dizinler ===
data_dir = r"C:\Users\tunah\Desktop\Brain_Cancer_Project\Brain_Cancer"
save_dir = r"C:\Users\tunah\models"
os.makedirs(save_dir, exist_ok=True)

# === Veri Ön İşleme ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),  # Xception genelde 299x299 ister, ama 224 de yeterli
    batch_size=32,
    subset='training',
    class_mode='categorical'
)

val_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=32,
    subset='validation',
    class_mode='categorical'
)

# === Xception Modeli ===
base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# 🔹 Fine-tuning ayarı
base_model.trainable = True
for layer in base_model.layers[:-40]:  # sadece son 40 katmanı eğit
    layer.trainable = False

# === Model Mimarisi ===
model_xcep = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

# === Derleme ===
optimizer = Adam(learning_rate=1e-4)
model_xcep.compile(optimizer=optimizer,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

model_xcep.summary()

# === Callback'ler ===
checkpoint = ModelCheckpoint(
    os.path.join(save_dir, "best_xception.keras"),
    monitor="val_accuracy", save_best_only=True, verbose=1
)
earlystop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
csv_logger = CSVLogger(os.path.join(save_dir, "xception_history.csv"))

# === Model Eğitimi ===
history_xcep = model_xcep.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[checkpoint, earlystop, csv_logger]
)

# === Accuracy / Loss Grafik ===
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_xcep.history['accuracy'], label='Eğitim')
plt.plot(history_xcep.history['val_accuracy'], label='Doğrulama')
plt.title('Xception - Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history_xcep.history['loss'], label='Eğitim')
plt.plot(history_xcep.history['val_loss'], label='Doğrulama')
plt.title('Xception - Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "xception_acc_loss.png"), dpi=300)
plt.show()

print(f"\n✅ Xception eğitimi tamamlandı! Sonuçlar kaydedildi: {save_dir}")
