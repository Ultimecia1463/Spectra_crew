import numpy as np
import matplotlib.pyplot as plt
import rasterio

# Corrected file paths with raw strings to handle backslashes correctly
img_file = 'D:\sih\Spectra_crew\\ang20180403t053539\\ang20180403t053539_rdn_v2s1\\ang20180403t053539_rdn_v2s1_img'
hdr_file = 'D:\sih\Spectra_crew\\ang20180403t053539\\ang20180403t053539_rdn_v2s1\\ang20180403t053539_rdn_v2s1_img.hdr'

# Check if the file paths are correct
print(f"Looking for image file at: {img_file}")

try:
    # Open the radiance image file using rasterio
    with rasterio.open(img_file) as dataset:
        # Read a specific band (e.g., Band 10, adjust as needed)
        band_index = 10  # Modify this index to visualize other bands
        band = dataset.read(band_index)

        # Print metadata for confirmation
        print("Band count:", dataset.count)
        print("Width:", dataset.width)
        print("Height:", dataset.height)
        print("Data type:", band.dtype)
except rasterio.errors.RasterioIOError as e:
    print(f"Error opening the file: {e}")
    exit()

# Normalize the data for better visualization
band = np.clip(band, np.percentile(band, 2), np.percentile(band, 98))  # Clipping outliers
band = (band - band.min()) / (band.max() - band.min())  # Normalize to 0-1

# Plot the image
plt.figure(figsize=(10, 8))
plt.imshow(band, cmap='viridis')  # Change the colormap as needed
plt.colorbar(label='Radiance')
plt.title(f'AVIRIS-NG Radiance Image - Band {band_index}')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()
