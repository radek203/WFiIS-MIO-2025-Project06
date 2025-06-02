import matplotlib.pyplot as plt
import numpy as np
import os

def draw_plots(history1, history2, name1, name2, name, output_dir='results/iris'):
    os.makedirs(output_dir, exist_ok=True)  
    plt.figure(figsize=(12, 18))

    plt.subplot(3, 1, 1)
    plt.plot(history1.history['loss'], label=f'{name1} Train Loss')
    plt.plot(history1.history['val_loss'], label=f'{name1} Test Loss')
    plt.plot(history2.history['loss'], label=f'{name2} Train Loss')
    plt.plot(history2.history['val_loss'], label=f'{name2} Test Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(history1.history['accuracy'], label=f'{name1} Train Accuracy')
    plt.plot(history1.history['val_accuracy'], label=f'{name1} Test Accuracy')
    plt.plot(history2.history['accuracy'], label=f'{name2} Train Accuracy')
    plt.plot(history2.history['val_accuracy'], label=f'{name2} Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(history1.history['f1_score'], label=f'{name1} Train F1')
    plt.plot(history1.history['val_f1_score'], label=f'{name1} Test F1')
    plt.plot(history2.history['f1_score'], label=f'{name2} Train F1')
    plt.plot(history2.history['val_f1_score'], label=f'{name2} Test F1')
    plt.title('F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"{name}_{name1}_{name2}.png")
    plt.savefig(filepath)
    plt.show()

def augment_data(X, noise_factor=0.05):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=X.shape)
    return X + noise