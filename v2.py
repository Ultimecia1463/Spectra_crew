import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import colour
from skimage.exposure import rescale_intensity
from Accessories import D65, xyz

# Define file paths
img_file = 'D:\sih\Spectra_crew\\ang20180403t053539\\ang20180403t053539_rdn_v2s1\\ang20180403t053539_rdn_v2s1_img'
hdr_file = 'D:\sih\Spectra_crew\\ang20180403t053539\\ang20180403t053539_rdn_v2s1\\ang20180403t053539_rdn_v2s1_img.hdr'


def read_hyperspectral_data_in_chunks(img_path, hdr_path, chunk_size=512):
    with open(hdr_path, 'r') as hdr:
        metadata = hdr.read()
    
    wavelengths = extract_wavelengths_from_hdr(metadata)
    
    with rasterio.open(img_path) as dataset:
        width = dataset.width
        height = dataset.height
        bands = dataset.count
        
        hyperspectral_data = []

        for i in range(0, height, chunk_size):
            row_chunks = []
            for j in range(0, width, chunk_size):
                window_height = min(chunk_size, height - i)
                window_width = min(chunk_size, width - j)
                window = rasterio.windows.Window(j, i, window_width, window_height)
                chunk = dataset.read(window=window)
                
                # Pad chunks to ensure consistent size
                padded_chunk = np.zeros((bands, chunk_size, chunk_size), dtype=chunk.dtype)
                padded_chunk[:, :window_height, :window_width] = chunk
                row_chunks.append(padded_chunk)

            # Concatenate row chunks along width
            row_concat = np.concatenate(row_chunks, axis=2)
            hyperspectral_data.append(row_concat)

        # Concatenate all rows along height
        hyperspectral_data = np.concatenate(hyperspectral_data, axis=1)
    
    return hyperspectral_data, wavelengths

def extract_wavelengths_from_hdr(metadata):
    # Implement extraction logic based on your HDR file format
    wavelengths = np.linspace(400, 700, 120)  # Example wavelength range
    return wavelengths

# Read hyperspectral data and metadata
hyperspectral_data, wavelengths = read_hyperspectral_data_in_chunks(img_file, hdr_file)

# Interpolation setup
wavelengths = wavelengths[0:120]  # Adjust to match your wavelengths if necessary
x = np.arange(400, 701, 5)
x = x.reshape((61, 1))
x = x.ravel()

# D65 standard illuminant
y = D65
y = y.ravel()
interp_function = interp1d(x, y, kind='linear', fill_value="extrapolate")
x_new = wavelengths
D65n = interp_function(x_new)

# Standard observer
y = xyz
interp_func_0 = interp1d(x, xyz[:, 0], kind='linear', fill_value='extrapolate')
interp_func_1 = interp1d(x, xyz[:, 1], kind='linear', fill_value='extrapolate')
interp_func_2 = interp1d(x, xyz[:, 2], kind='linear', fill_value='extrapolate')
interpolated_values_0 = interp_func_0(x_new)
interpolated_values_1 = interp_func_1(x_new)
interpolated_values_2 = interp_func_2(x_new)
xyzn = np.column_stack((interpolated_values_0, interpolated_values_1, interpolated_values_2))

# Convert reflectance to CIEXYZ tristimulus values
RI_Slotff = hyperspectral_data[:, :, 0:120]
RI_Slotff = RI_Slotff.reshape((670*1062, 120))
Mul_temp = np.matmul(np.transpose(xyzn), np.diagflat(D65n))
Mul_temp2 = np.matmul(Mul_temp, np.transpose(RI_Slotff))
XYZ = 1/100 * Mul_temp2

# XYZ to sRGB
XYZ = (np.transpose(XYZ)).reshape((670, 1062, 3))
SRGB = colour.XYZ_to_sRGB(XYZ)

# Apply the contrast stretch
SRGB = rescale_intensity(SRGB)

# Display the image
plt.imshow(SRGB)
plt.title('Colorimetric Visualization of Hyperspectral Data')
plt.axis('off')  # Hide axis
plt.show()
