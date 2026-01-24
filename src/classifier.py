import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import time
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# for reproducibility
random_state=123
tf.random.set_seed(123)
tf.keras.utils.set_random_seed(123)
tf.config.experimental.enable_op_determinism()

def download_dataset(dataset_name):
    """Download and cache datasets."""
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)
    
    if dataset_name == "cifar10":
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
        input_shape = (32, 32, 3)
        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                      'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        cmap = None
    else:  # fashion_mnist
        (X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
        input_shape = (28, 28, 1)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        cmap = "gray"
    
    # Normalize
    if dataset_name == "fashion_mnist":
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    else:
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test, input_shape, class_names, cmap

def build_model(input_shape):
    model = models.Sequential([

        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation="relu"), layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    return model

def main():
    parser = argparse.ArgumentParser(description="Image Classifier: CIFAR-10 or Fashion-MNIST")
    parser.add_argument("--dataset", choices=["cifar10", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--model-path", default="models/CNNmodel_.keras", help="Path to save/load model")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Operation mode")
    args = parser.parse_args()
    
    print(f"Using dataset: {args.dataset}")

    if args.dataset == "cifar10":
        args.model_path = args.model_path.replace(".keras", "cifar10.keras")
    else:
        args.model_path = args.model_path.replace(".keras", "fashion.keras")
    
    # Load data
    X_train, y_train, X_test, y_test, input_shape, class_names, cmap = download_dataset(args.dataset)
    
    model = build_model(input_shape)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    if args.mode == "train":
        startt = time.time()
        history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2, verbose=1)
        model.save(args.model_path)
        endt = time.time()
        tm = endt - startt
        print(f"Model saved to {args.model_path}")
        print(f"Training duration: {tm}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()
        
    elif args.mode == "test":
        index = 42
        model = tf.keras.models.load_model(args.model_path)
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose = 0)

        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        y_true = np.argmax(y_test, axis=1)

        startp = time.time()
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        endp = time.time()

        cm = confusion_matrix(y_true, y_pred)

        img = X_test[index]

        true_label = class_names[np.argmax(y_test[index])]
        pred_probs = model.predict(np.expand_dims(img, axis=0), verbose = 0)
        pred_label = class_names[np.argmax(pred_probs)]

        plt.figure()
        plt.imshow(img, cmap=cmap)
        plt.title(f"Truth: {true_label}\nPredicted: {pred_label}")
        plt.axis("off")
        plt.show()
        
        print(cm)
        print (f"Prediction Duration: {(endp - startp):.4f}")

if __name__ == "__main__":
    main()