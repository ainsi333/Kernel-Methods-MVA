import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("data loading")
Xtr = pd.read_csv('Xtr.csv', header=None, usecols=range(3072)).values
Ytr = pd.read_csv('Ytr.csv', usecols=[1]).values.squeeze()

def reshape_and_normalize(X):
    X_img = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    #min-max normalization per image
    v_min = X_img.min(axis=(1, 2, 3), keepdims=True)
    v_max = X_img.max(axis=(1, 2, 3), keepdims=True)

    #to prevent division by zero for flat images :
    diff = v_max - v_min
    diff[diff == 0] = 1 
    
    return (X_img - v_min) / diff

X_vis = reshape_and_normalize(Xtr)

#display a random grid of 25 images with their labels :
class_names = {
    0: 'Avion', 1: 'Auto', 2: 'Oiseau', 3: 'Chat', 4: 'Cerf',
    5: 'Chien', 6: 'Grenouille', 7: 'Cheval', 8: 'Bateau', 9: 'Camion'
}

fig, axes = plt.subplots(5, 5, figsize=(10, 10))

indices = np.random.choice(len(Xtr), 25, replace=False)

for ax, idx in zip(axes.flat, indices):
    label = int(Ytr[idx])
    ax.imshow(X_vis[idx])
    ax.set_title(f"{class_names.get(label, 'Inconnu')} ({label})")
    ax.axis('off')

plt.tight_layout()
plt.show()
