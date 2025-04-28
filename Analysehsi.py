from spectral import open_image
import numpy as np
from skimage import filters, measure, morphology
import pandas as pd
import matplotlib.pyplot as plt

# Load HSI image
hsi_image = open_image('partial-dataset/Data-VIS-20170117-1-room-light-off/CS6-01.hdr')
hsi_data = hsi_image.load()  # Shape: (height, width, bands)

# Preprocess: Compute mean reflectance
hsi_mean = np.mean(hsi_data, axis=2)

# Segment grains
thresh = filters.threshold_otsu(hsi_mean)
binary = hsi_mean > thresh
binary = morphology.remove_small_objects(binary, min_size=50)
binary = morphology.binary_closing(binary, morphology.disk(3))
labeled_image = measure.label(binary, connectivity=2)
regions = measure.regionprops(labeled_image)

# Extract spectral data for each grain
grain_data = []
for region in regions:
    coords = region.coords
    spectra = hsi_data[coords[:, 0], coords[:, 1], :]
    mean_spectrum = np.mean(spectra, axis=0)
    grain_info = {
        'grain_id': region.label,
        'area': region.area,
        **{f'band_{i+1}': mean_spectrum[i] for i in range(mean_spectrum.shape[0])}
    }
    grain_data.append(grain_info)

# Save to CSV
df = pd.DataFrame(grain_data)
df.to_csv('grain_data.csv', index=False)

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(hsi_mean, cmap='gray')
plt.title('Mean Reflectance')
plt.subplot(1, 2, 2)
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.title('Segmented Grains')
plt.show()