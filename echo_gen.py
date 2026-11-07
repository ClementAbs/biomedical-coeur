"""Générateur d'images pseudo-échographiques du ventricule gauche (vue simplifiée)."""
import numpy as np
from scipy.ndimage import gaussian_filter

def make_echo(rng, pathologique, size=64):
    """Une image 64x64 : cavité sombre entourée d'un muscle (myocarde) plus clair, bruit de speckle."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx = size/2 + rng.uniform(-3, 3)
    cy = size/2 + rng.uniform(-3, 3)
    ang = rng.uniform(-0.4, 0.4)
    if pathologique:      # cavité dilatée + paroi plus fine
        a, b = rng.uniform(12, 18), rng.uniform(16, 22)
        wall = rng.uniform(3, 6)
    else:                 # cœur normal
        a, b = rng.uniform(8, 13), rng.uniform(12, 17)
        wall = rng.uniform(5.5, 9)
    x, y = xx - cx, yy - cy
    xr = x*np.cos(ang) + y*np.sin(ang)
    yr = -x*np.sin(ang) + y*np.cos(ang)
    r_in = (xr/a)**2 + (yr/b)**2
    r_out = (xr/(a+wall))**2 + (yr/(b+wall))**2
    img = np.full((size, size), 0.15)
    img[r_out <= 1] = 0.65            # myocarde
    img[r_in <= 1] = 0.08             # sang (sombre)
    img = gaussian_filter(img, 1.2)
    speckle = rng.rayleigh(scale=1.0, size=(size, size))   # bruit multiplicatif typique des ultrasons
    speckle = gaussian_filter(speckle, 0.8)
    img = img * (0.55 + 0.6*speckle) * rng.uniform(0.8, 1.2)   # gain aléatoire
    img += rng.normal(0, 0.03, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)

def make_dataset(n=600, prevalence=0.3, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    X = np.stack([make_echo(rng, bool(v)) for v in y])
    return X, y
