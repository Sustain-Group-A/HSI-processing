import matplotlib.pyplot as plt


#code to plot the mean reflectance of each grain type and show the segmented grains
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(hsi_mean, cmap='gray')
plt.title('Mean Reflectance')
plt.subplot(1, 2, 2)
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.title('Segmented Grains')
plt.show()