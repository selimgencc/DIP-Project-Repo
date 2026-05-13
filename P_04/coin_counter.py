import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


def goster(img, baslik, gri=False):
    plt.figure(figsize=(6, 5))
    if gri:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(baslik)
    plt.axis('off')
    plt.show()


# 1) RGB oku
img_bgr = cv2.imread("I.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
goster(img_rgb, "1 - Orijinal RGB Görüntü ")

# 2) Medyan filtre (RGB) -> tuz-biber gurultusunu kaldir
filt_bgr = cv2.medianBlur(img_bgr, 5)
filt_rgb = cv2.cvtColor(filt_bgr, cv2.COLOR_BGR2RGB)
goster(filt_rgb, "2 - Medyan Filtre Uygulandı")

# 3) Grayscale
gray = cv2.cvtColor(filt_bgr, cv2.COLOR_BGR2GRAY)
goster(gray, "3 - Grayscale'e geçildi", gri=True)

# 4) Otsu inverse threshold -> binary
_, binary = cv2.threshold(gray, 0, 255,
cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
goster(binary, "4 - Otsu Inverse Threshold Uygulandı", gri=True)

# 5) Circularity filtresi -> dartlari ele
label_bin = measure.label(binary, connectivity=2)
props = measure.regionprops(label_bin)
coins_only = np.zeros_like(binary)

for p in props:
    perim = p.perimeter if p.perimeter > 0 else 1
    circ = 4 * np.pi * p.area / (perim ** 2)
    if circ > 0.5 and p.area > 300:
        coins_only[label_bin == p.label] = 255
goster(coins_only, "5) Daire Filtresi Uygulandı, Dartlar gitti", gri=True)

# 6) Distance Transform -> topografik yuzey
dist = cv2.distanceTransform(coins_only, cv2.DIST_L2, 5)
goster(dist, "6) Distance Transform Uygulandı", gri=True)

# 7) DT yerel maksimumlari -> markers (her tepe = bir para tohumu)
coords = peak_local_max(dist, min_distance=20)
mask_peaks = np.zeros(dist.shape, dtype=bool)
mask_peaks[tuple(coords.T)] = True
markers, num_markers = ndi.label(mask_peaks)
goster(markers, "7) Markers (DT yerel maksimumlari, tek piksel peak bulundu)")

# 8) Watershed -> -DT uzerinde, markers ile
labels = watershed(-dist, markers, mask=(dist > 0))
goster(labels, "8) Watershed Etiketleri Uygulandı")

# 9) Say + gorsellestir
coin_count = labels.max()

result_img = img_rgb.copy()
for label_id in range(1, coin_count + 1):
    component = (labels == label_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours, -1, (255, 0, 0), 2)

goster(result_img, f"9) Sonuc: {coin_count} adet 1 TL")
print(f"\n>>> Bulunan 1 TL sayisi: {coin_count}")
