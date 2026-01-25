import argparse
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import pickle
from pathlib import Path
import cv2
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import datasets
import seaborn as sns

# for reproducibility
random_state=123
tf.random.set_seed(123)
tf.keras.utils.set_random_seed(123)


def download_dataset(dataset_name):
    if dataset_name == "cifar10":
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
        class_names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
        cmap = None
    else:  # fashion_mnist
        (X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
        cmap = "gray"
    
    return X_train, y_train, X_test, y_test, class_names, cmap

def extract_histogram_features(X, dataset_name, bins=256):
    features = []
    if dataset_name.lower() == 'cifar10':
        for img in X:
            imagePIL = Image.fromarray(img)
            featureVector=imagePIL.histogram()
            if (len(featureVector) != 768): # just a sanity check; 3 * 256 for RGB
                print ("Unexpected length of feature vector: " + str(len(featureVector)) + " in file: " + img)
            features.append((featureVector))
            
    else:  # fashion_mnist
        for img in X:
            imagePIL = Image.fromarray(img)
            featureVector=imagePIL.histogram()
            if (len(featureVector) != 256): # just a sanity check; 256 for greyscale
                print ("Unexpected length of feature vector: " + str(len(featureVector)) + " in file: " + img)
            features.append((featureVector))
    
    X_hist = np.array(features)
    return X_hist

def plot_sample_image(X_train, X_train_hist, y_train, dataset_name, index=0):
    img = X_train[index]
    label = y_train[index]

    plt.figure(figsize=(10, 3))
    # Plot image
    plt.subplot(1, 2, 1)
    if dataset_name.lower() == 'fashion_mnist':
        plt.imshow(img, cmap='gray')  # Grayscale
    else:
        
        plt.imshow(img) # RGB

        
    plt.title(f'{dataset_name.title()} Image {index}\nClass: {label}')
    plt.axis('off')
    featureVector=X_train_hist[index]

    
    # Plot histogram (already dynamic)
    plt.subplot(1, 2, 2)
    hist = extract_histogram_features(np.array([img]), dataset_name)[0]
    plt.plot(hist, linewidth=2)
    plt.title(f'{dataset_name.title()} Histogram')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.1)
    plt.plot(featureVector[:256], 'r')
    plt.plot(featureVector[257:512], 'g')
    plt.plot(featureVector[513:], 'b')
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()

def visualize_sift_keypoints(img):
    """Universal: handles grayscale (2D) and RGB (3D) correctly"""
    sift = cv2.SIFT_create()
    
    # Prepare grayscale input (handle both cases)
    if len(img.shape) == 3:  # RGB CIFAR
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:  # Grayscale Fashion-MNIST (28,28)
        gray = (img * 255).astype(np.uint8)  # Already grayscale!
    
    kp, _ = sift.detectAndCompute(gray, None)
    img_kp = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.figure(figsize=(12, 5))
    
    # Original
    plt.subplot(1, 2, 1)
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Keypoints (always grayscale output)
    plt.subplot(1, 2, 2)
    plt.imshow(img_kp, cmap='gray')
    plt.title(f'SIFT Keypoints ({len(kp)} detected)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_bovw_histogram(hist, title='BoVW Histogram'):
    """Plot normalized word histogram"""
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(hist)), hist)
    plt.title(title)
    plt.xlabel('Visual Word ID')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

def extract_sift_descriptors(X):
    """Universal: handles grayscale + RGB"""
    sift = cv2.SIFT_create()
    all_descriptors = []
    
    for img in X:
        # Handle both datasets
        if len(img.shape) == 3:  # CIFAR RGB
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:  # Fashion grayscale
            gray = (img * 255).astype(np.uint8)
        
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is not None:
            all_descriptors.append(desc)
    
    descriptors = np.vstack(all_descriptors)
    print(f"Total descriptors: {descriptors.shape[0]} (dim: {descriptors.shape[1]})")
    return descriptors

def build_codebook(descriptors, n_words=256):
    """KMeans clustering → visual vocabulary"""
    kmeans = MiniBatchKMeans(n_clusters=n_words, random_state=42)
    kmeans.fit(descriptors)
    return kmeans

def image_bovw(img, codebook):
    """Fixed single image BoVW"""
    sift = cv2.SIFT_create()
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (img * 255).astype(np.uint8)
    
    kp, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return np.zeros(len(codebook.cluster_centers_))
    
    words = codebook.predict(desc)
    hist, _ = np.histogram(words, bins=len(codebook.cluster_centers_), range=(0, len(codebook.cluster_centers_)))
    return hist / hist.sum()

def extract_bovw_features(X, codebook):
    """BoVW for entire dataset"""
    return np.array([image_bovw(img, codebook) for img in X])

def visualize_codebook(codebook, n_display=16):
    """Show first n visual words (cluster centers)"""
    centers = codebook.cluster_centers_.reshape(-1, 8, 8)  # Assuming 64-dim SIFT patches
    plt.figure(figsize=(8, 8))
    for i in range(min(n_display, len(centers))):
        plt.subplot(4, 4, i+1)
        plt.imshow(centers[i], cmap='gray')
        plt.title(f'Word {i}')
        plt.axis('off')
    plt.suptitle('Visual Codebook (First 16 Words)')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Image Classifier: CIFAR-10 or Fashion-MNIST")
    parser.add_argument("--dataset", choices=["cifar10", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument("--model_type", choices=["HIST", "SIFT"], default="HIST", help="Model architecture (Hist, SIFT, CNN or RNN)")
    parser.add_argument("--model-path", default="models/histmodel_.pkl", help="Path to save/load model")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Operation mode")
    args = parser.parse_args()
    
    print(f"Using dataset: {args.dataset}")
    print(f"Model type: {args.model_type}")
    model_filename = f"{args.model_type}model_{args.dataset}.pkl"
    args.model_path = f"models/{model_filename}"
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test, class_names, cmap = download_dataset(args.dataset)

    if args.model_type == "SIFT":
        visualize_sift_keypoints(X_train[0])

        print("Extracting SIFT descriptors...")
        train_descriptors = extract_sift_descriptors(X_train)
        codebook = build_codebook(train_descriptors, n_words=128)
        visualize_codebook(codebook)

        print("Extracting BoVW...")
        X_train_bovw = extract_bovw_features(X_train, codebook)
        sample_hist = X_train_bovw[123]
        visualize_bovw_histogram(sample_hist, 'Sample Image BoVW Histogram')

    elif args.model_type == "HIST":
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        start = time.time()
        X_train_hist = extract_histogram_features(X_train, args.dataset)
        end = time.time()
        print (f"Feature Extraction Duration - Train set: {(end - start):.4f}")

        start = time.time()
        X_test_hist = extract_histogram_features(X_test, args.dataset)
        end = time.time()
        print (f"Feature Extraction Duration - Test set: {(end - start):.4f}")
        
        plot_sample_image(X_train, X_train_hist, y_train, args.dataset)

        X_train = X_train_hist
        X_test = X_test_hist
    
    if args.mode == "train":
        pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf'))])
        print("Start Model Training")
        startt = time.time()
        pipeline.fit(X_train, y_train)
        endt = time.time()
        tm = endt - startt
        print(f"Training duration: {tm}")
        with open(args.model_path,'wb') as f:
            pickle.dump(pipeline,f)
            print(f"Model saved to {args.model_path}")


    elif args.mode == "test":
        print(f"Model: {model_filename}")
        index = 123
        with open(args.model_path, 'rb') as f:
            pipelineT = pickle.load(f)

        print("Start Model Testing")
        start = time.time()
        accuracy = pipelineT.score(X_test, y_test)
        end = time.time()
        print (f"Evaluation - Test set - Duration: {(end - start):.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        y_true = np.argmax(y_test, axis=1)

        print("Start Model Predictions")

        startp = time.time()
        y_pred = pipelineT.predict(X_test, verbose=0).argmax(axis=1)
        endp = time.time()

        cm = confusion_matrix(y_true, y_pred)

        img = X_test[index]
        true_label = class_names[np.argmax(y_test[index])]

        pred_probs = pipelineT.predict(np.expand_dims(img, axis=0), verbose = 0)
        pred_label = class_names[np.argmax(pred_probs)]

        plt.figure()
        plt.imshow(img, cmap=cmap)
        plt.title(f"Sample Image: Truth: {true_label}\nPredicted: {pred_label}")
        plt.axis("off")
        plt.show()

        print(cm)
        print (f"Prediction Duration: {(endp - startp):.4f}")

        # 4. Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.title(f'Confusion Matrix: {args.model_type} ({args.dataset})')
        
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{args.model_type}_{args.dataset}_cm.png")
        plt.show()

if __name__ == "__main__":
    main()