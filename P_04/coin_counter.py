import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

IMG_PATH = "I.png"
MEDIAN_KSIZE = 5
MIN_CIRCULARITY = 0.5
MIN_AREA = 300
PEAK_MIN_DISTANCE = 20
steps = []

def show(img, title, gray=False):
    # Collect the step instead of displaying it immediately
    steps.append((img, title, gray))

def show_all():
    n = len(steps)
    idx = [0]
    fig, ax = plt.subplots(figsize=(7, 6))

    def draw():
        ax.clear()
        img, title, gray = steps[idx[0]]
        ax.imshow(img, cmap='gray') if gray else ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ('right', ' ', 'enter'):
            idx[0] = (idx[0] + 1) % n
            draw()
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % n
            draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw()
    plt.show()

# Read RGB
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"{IMG_PATH} not found")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
show(img_rgb, "Original RGB Image")

# Median filter (RGB) -> remove salt-and-pepper noise
filt_bgr = cv2.medianBlur(img_bgr, MEDIAN_KSIZE)
filt_rgb = cv2.cvtColor(filt_bgr, cv2.COLOR_BGR2RGB)
show(filt_rgb, "Median Filter Applied")

# Grayscale
gray = cv2.cvtColor(filt_bgr, cv2.COLOR_BGR2GRAY)
show(gray, "Grayscale", gray=True)

# Otsu inverse threshold -> binary
_, binary = cv2.threshold(gray, 0, 255,
cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
show(binary, "Otsu Inverse Threshold", gray=True)

# Circularity filter -> discard the darts
label_bin = measure.label(binary, connectivity=2)
props = measure.regionprops(label_bin)
coins_only = np.zeros_like(binary)

for p in props:
    perim = p.perimeter if p.perimeter > 0 else 1
    circ = 4 * np.pi * p.area / (perim ** 2)
    if circ > MIN_CIRCULARITY and p.area > MIN_AREA:
        coins_only[label_bin == p.label] = 255
show(coins_only, "Circle Filter Applied (darts removed)", gray=True)

# Distance Transform -> topographic surface
dist = cv2.distanceTransform(coins_only, cv2.DIST_L2, 5)
show(dist, "Distance Transform", gray=True)

# Local maxima of DT -> markers (each peak = one coin seed)
coords = peak_local_max(dist, min_distance=PEAK_MIN_DISTANCE)
mask_peaks = np.zeros(dist.shape, dtype=bool)
mask_peaks[tuple(coords.T)] = True
markers, _ = ndi.label(mask_peaks)
show(markers, "Markers (DT local maxima)")

# Watershed -> on -DT, with markers
labels = watershed(-dist, markers, mask=(dist > 0))
show(labels, "Watershed Labels")

# Count + visualize
coin_count = labels.max()

result_img = img_rgb.copy()
for label_id in range(1, coin_count + 1):
    component = (labels == label_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours, -1, (255, 0, 0), 2)

show(result_img, f"Result: {coin_count} x 1 TL")

# Display all steps at the end
show_all()
print(f"\n>>> Number of 1 TL coins found: {coin_count}")