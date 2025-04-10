import tensorflow as tf
import numpy as np
from datasets import load_dataset
import cv2
import os

def load_and_preprocess_dataset():
    
    ds = load_dataset("garythung/trashnet")


    def preprocess_data(example):
        image = np.array(example["image"])
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        label = tf.one_hot(example["label"], depth=6)
        return image, label

   
    train_data = tf.data.Dataset.from_generator(
        lambda: (preprocess_data(example) for example in ds["train"]),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(6,), dtype=tf.float32),
        )
    )

    
    train_data = train_data.batch(32).shuffle(100).prefetch(tf.data.AUTOTUNE)
    return train_data


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(6, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, train_data):
    model.fit(train_data, epochs=30)
    return model


def save_model_h5(model, path="trashnet_detection.h5"):
    model.save(path)
    print(f"Model '{path}' dosyasına kaydedildi.")


def load_model_h5(path="trashnet_detection.h5"):
    if os.path.exists(path):
        print(f"Model '{path}' dosyasından yüklendi.")
        return tf.keras.models.load_model(path)
    else:
        raise FileNotFoundError(f"Model '{path}' bulunamadı. Lütfen önce modeli eğitin ve kaydedin.")


def detect_with_camera(model):
    cap = cv2.VideoCapture(0)  

    classes = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        image = cv2.resize(frame, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        
        
        predictions = model.predict(image)
        class_id = np.argmax(predictions)
        
        
        label = classes[class_id]
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Trash Detection", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "trashnet_detection.h5"
    if os.path.exists(model_path):
        print("Eğitilmiş model yükleniyor...")
        model = load_model_h5(model_path)
    else:
        print("Model eğitiliyor...")
        train_data = load_and_preprocess_dataset()
        model = create_model()
        model = train_model(model, train_data)
        save_model_h5(model, model_path)
    
    print("Kamera başlatılıyor...")
    detect_with_camera(model)
