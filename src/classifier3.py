
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns 
import pandas as pd

# For reproducibility
random_state = 123
tf.random.set_seed(123)
tf.keras.utils.set_random_seed(123)
tf.config.experimental.enable_op_determinism()

# --- ViT HELPER LAYERS (FIXED) ---

class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    # def call(self, patch):
    #     positions = tf.range(start=0, limit=self.num_patches, delta=1)
    #     encoded = self.projection(patch) + self.position_embedding(positions)
    #     return encoded
    def call(self, patch):
        batch_size = tf.shape(patch)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # This ensures the position embedding is added correctly to every image in the batch
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

# --- CORE FUNCTIONS ---

def download_dataset(dataset_name, model_type):
    if dataset_name == "cifar10":
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
        img_rows, img_cols, channels = 32, 32, 3
        class_names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
        cmap = None
    else:  # fashion_mnist
        (X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
        img_rows, img_cols, channels = 28, 28, 1
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
        cmap = "gray"

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    if model_type in ["cnn", "vit"]:
        X_train = X_train.reshape((X_train.shape[0], img_rows, img_cols, channels))
        X_test = X_test.reshape((X_test.shape[0], img_rows, img_cols, channels))
        input_shape = (img_rows, img_cols, channels)
    else:  # rnn
        y_train = np.squeeze(y_train).ravel()
        y_test = np.squeeze(y_test).ravel()
        X_train = X_train.reshape((X_train.shape[0], img_rows, img_cols * channels))
        X_test = X_test.reshape((X_test.shape[0], img_rows, img_cols * channels))
        input_shape = (img_rows, img_cols * channels)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test, input_shape, class_names, cmap

def build_model(model_type, input_shape):
    if model_type == "cnn":
        return models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
            layers.Conv2D(64, (3,3), activation='relu'), 
            layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
            layers.Flatten(), layers.Dense(128, activation='relu'),
            layers.Dropout(0.5), layers.Dense(10, activation='softmax')
        ])
    
    elif model_type == "vit":
        patch_size = 4 if input_shape[0] == 32 else 7 
        num_patches = (input_shape[0] // patch_size) ** 2
        projection_dim = 128
        num_heads = 4
        transformer_layers = 4
        
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        inputs = layers.Input(shape=input_shape)
        augmented = data_augmentation(inputs)
        patches = PatchExtract(patch_size)(augmented)
        encoded_patches = PatchEmbedding(num_patches, projection_dim)(patches)

        for _ in range(transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = layers.Dense(projection_dim * 4, activation=tf.nn.gelu)(x3)
            x3 = layers.Dropout(0.1)(x3)
            x3 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)
            encoded_patches = layers.Add()([x3, x2])

        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.GlobalAveragePooling1D()(representation) 
        representation = layers.Dropout(0.3)(representation)
        features = layers.Dense(128, activation=tf.nn.gelu)(representation)
        logits = layers.Dense(10, activation="softmax")(features)
        
        return tf.keras.Model(inputs=inputs, outputs=logits)

    else:  # rnn
        return models.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

def main():
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Image Classifier")
    parser.add_argument("--dataset", choices=["cifar10", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("--model_type", choices=["cnn", "rnn", "vit"], default="cnn")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    model_filename = "{}model_{}.keras".format(args.model_type, args.dataset)
    model_path = os.path.join("models", model_filename)

    X_train, y_train, X_test, y_test, input_shape, class_names, cmap = download_dataset(args.dataset, args.model_type)
    
    if args.mode == "train":
        model = build_model(args.model_type, input_shape)
        
        if args.model_type == "vit":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.0001)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        
        start = time.time()
        history = model.fit(X_train, y_train, epochs=args.epochs, 
                            batch_size=args.batch_size, validation_split=0.2)
        model.save(model_path)
        print(f"Training Duration: {time.time() - start:.2f}s")   
        # --- FIXED PLOTTING LOGIC ---
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='val')
        plt.title(f'Training Accuracy: {args.model_type} on {args.dataset}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        train_plot_path = f"plots/{args.model_type}_{args.dataset}_training.png"
        plt.savefig(train_plot_path)
        print(f"Training plot saved to: {train_plot_path}")
        plt.show()
        plt.close()
 
    elif args.mode == "test":
     
        print(f"Loading Deep Learning Model: {model_path}")
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'PatchExtract': PatchExtract, 'PatchEmbedding': PatchEmbedding}
        )

        print("Predicting on Test Set...")
        # 1. Measure Latency
        start_time = time.time()
        y_pred_probs = model.predict(X_test, verbose=0)
        end_time = time.time()

        # 2. Convert from Probs/One-Hot to 1D integers
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        avg_latency = (end_time - start_time) / len(X_test)
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        # 3. Print & Save Results
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

        os.makedirs("results", exist_ok=True)
        results_path = "results/experiment_summary.csv"
        new_result = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Dataset": args.dataset,
            "Model": args.model_type,
            "Accuracy": round(acc, 4),
            "F1_Macro": round(f1_macro, 4),
            "Latency_ms": round(avg_latency * 1000, 4),
            "Size_MB": round(model_size_mb, 2)
        }
        pd.DataFrame([new_result]).to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)

        # 4. Heatmap
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.title(f'Confusion Matrix: {args.model_type} ({args.dataset})')
        
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{args.model_type}_{args.dataset}_cm.png")
        plt.show()

if __name__ == "__main__":
    main()