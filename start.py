import os
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from tqdm import tqdm

np.random.seed(42)

class HOGExtractor:
    def __init__(self, cell_size=4, n_bins=9):
        self.cell_size = cell_size
        self.n_bins = n_bins

    def compute(self, X_batch):
        n_samples = X_batch.shape[0]
        X_r = X_batch[:, :1024].reshape(n_samples, 32, 32)
        X_g = X_batch[:, 1024:2048].reshape(n_samples, 32, 32)
        X_b = X_batch[:, 2048:].reshape(n_samples, 32, 32)
        imgs = 0.299 * X_r + 0.587 * X_g + 0.114 * X_b

        features = []
        for i in tqdm(range(n_samples), desc="HOG", unit="img"):
            gx = np.zeros_like(imgs[i])
            gy = np.zeros_like(imgs[i])
            gx[:, 1:-1] = imgs[i, :, 2:] - imgs[i, :, :-2]
            gy[1:-1, :] = imgs[i, 2:, :] - imgs[i, :-2, :]
            
            mag = np.sqrt(gx**2 + gy**2)
            ang = (np.arctan2(gy, gx) * 180 / np.pi) % 180
            
            feat_vec = []
            for y in range(0, 32, self.cell_size):
                for x in range(0, 32, self.cell_size):
                    c_mag = mag[y:y+self.cell_size, x:x+self.cell_size]
                    c_ang = ang[y:y+self.cell_size, x:x+self.cell_size]
                    hist, _ = np.histogram(c_ang, bins=self.n_bins, range=(0, 180), weights=c_mag)
                    feat_vec.extend(hist)
            
            feat_arr = np.array(feat_vec)
            norm = np.linalg.norm(feat_arr) + 1e-6
            features.append(feat_arr / norm)
            
        return np.array(features)

class ColorHistExtractor:
    def __init__(self, n_bins=32):
        self.n_bins = n_bins

    def compute(self, X_batch):
        n_samples = X_batch.shape[0]
        features = []
        for i in tqdm(range(n_samples), desc="Color", unit="img"):
            h_r, _ = np.histogram(X_batch[i, :1024], bins=self.n_bins)
            h_g, _ = np.histogram(X_batch[i, 1024:2048], bins=self.n_bins)
            h_b, _ = np.histogram(X_batch[i, 2048:], bins=self.n_bins)
            hist = np.concatenate([h_r, h_g, h_b])
            features.append(hist / (np.sum(hist) + 1e-8))
        return np.array(features)

def augment_data(X, Y):
    X_img = X.reshape(-1, 3, 32, 32)
    n_base = X_img.shape[0]
    
    X_aug_list = [X]
    Y_aug_list = [Y]
    
    X_flip = np.flip(X_img, axis=3).reshape(n_base, -1)
    X_aug_list.append(X_flip)
    Y_aug_list.append(Y)
    
    X_trans = np.zeros_like(X_img)
    shifts = [-2, -1, 1, 2]
    for i in range(n_base):
        dx = np.random.choice(shifts)
        dy = np.random.choice(shifts)
        img = np.roll(X_img[i], shift=dy, axis=1)
        X_trans[i] = np.roll(img, shift=dx, axis=2)
    X_aug_list.append(X_trans.reshape(n_base, -1))
    Y_aug_list.append(Y)
    
    X_rot = np.zeros_like(X_img)
    for i in range(n_base):
        angle = np.random.uniform(-12, 12)
        X_rot[i] = rotate(X_img[i], angle, axes=(1, 2), reshape=False, mode='reflect')
    X_aug_list.append(X_rot.reshape(n_base, -1))
    Y_aug_list.append(Y)
    
    return np.concatenate(X_aug_list), np.concatenate(Y_aug_list)

def f_chi2(X1, X2, disable_tqdm=False, desc="Chi2"):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    dist = np.zeros((n1, n2))
    eps = 1e-10
    
    for i in tqdm(range(n1), desc=desc, unit="row", disable=disable_tqdm):
        diff_sq = (X1[i] - X2) ** 2
        sum_val = (X1[i] + X2) + eps
        dist[i, :] = 0.5 * np.sum(diff_sq / sum_val, axis=1)
        
    return dist

def solve_krr(K, Y, lambd):
    n = K.shape[0]
    Y_oh = np.zeros((n, 10))
    Y_oh[np.arange(n), Y.astype(int)] = 1
    return np.linalg.solve(K + lambd * np.eye(n), Y_oh)


if __name__ == "__main__":
    print("data loading")
    Xtr_raw = np.array(pd.read_csv('Xtr.csv', header=None, usecols=range(3072))) / 255.0
    Xte_raw = np.array(pd.read_csv('Xte.csv', header=None, usecols=range(3072))) / 255.0
    Ytr = np.array(pd.read_csv('Ytr.csv', usecols=[1])).squeeze()

    print("data augmentation")
    Xtr_aug, Ytr_aug = augment_data(Xtr_raw, Ytr)

    print("features extraction")
    hog = HOGExtractor(cell_size=4, n_bins=9)
    col = ColorHistExtractor(n_bins=32)
    
    Xtr_final = np.hstack([hog.compute(Xtr_aug), col.compute(Xtr_aug)])
    Xte_final = np.hstack([hog.compute(Xte_raw), col.compute(Xte_raw)])

    print("distance matrix calculation")
    D_full = f_chi2(Xtr_final, Xtr_final, desc="Train Dist")
  
    print("train/val splits")
    val_base_idx = np.random.choice(5000, 1000, replace=False)
    train_base_idx = np.setdiff1d(np.arange(5000), val_base_idx)
    
    train_idx = np.concatenate([
        train_base_idx,               
        train_base_idx + 5000,        
        train_base_idx + 10000,       
        train_base_idx + 15000        
    ])
    val_idx = val_base_idx 
    
    D_tr_tr = D_full[np.ix_(train_idx, train_idx)]
    D_val_tr = D_full[np.ix_(val_idx, train_idx)]
    
    Y_tr_s = Ytr_aug[train_idx]
    Y_val_s = Ytr_aug[val_idx]

    print("running validation")
    search_sigmas = [1.4028]
    best_acc = 0
    best_sigma = search_sigmas[0]
    lambd_cv = 0.0001
    
    for s in search_sigmas:
        K_tr = np.exp(-D_tr_tr / (2 * s**2))
        K_val = np.exp(-D_val_tr / (2 * s**2))
        
        alpha = solve_krr(K_tr, Y_tr_s, lambd_cv)
        preds = np.argmax(np.dot(K_val, alpha), axis=1)
        acc = np.mean(preds == Y_val_s)
        
        print(f"sigma={s:.4f} | val acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_sigma = s

    print("training of the final model")
    K_full = np.exp(-D_full / (2 * best_sigma**2))
    alpha_final = solve_krr(K_full, Ytr_aug, lambd=0.5)
    
    print("prediction on test set")
    D_test = f_chi2(Xte_final, Xtr_final, desc="Test Dist")
    K_test = np.exp(-D_test / (2 * best_sigma**2))
    preds_final = np.argmax(np.dot(K_test, alpha_final), axis=1)

    filename = f'Yte_pred.csv'
    df = pd.DataFrame({
        'Id': np.arange(1, len(preds_final) + 1),
        'Prediction': preds_final
    })
    df.to_csv(filename, index=False)