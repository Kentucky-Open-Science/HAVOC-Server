import json
import numpy as np
import matplotlib.pyplot as plt
import base64
import gzip
from io import BytesIO

def meters_to_rotated_pixel(x_meter, y_meter, origin_x, origin_y, resolution, top_crop_pixels=0, left_crop_pixels=0):
    # Debug print statements
    print(f"Converting meters ({x_meter}, {y_meter}) to pixel indices with origin ({origin_x}, {origin_y}) and resolution {resolution}")
    print(f"Top crop pixels: {top_crop_pixels}, Left crop pixels: {left_crop_pixels}")
    # Convert meters to pixel indices in the original image
    i = int(round((x_meter - origin_x) / resolution))
    j = int(round((y_meter - origin_y) / resolution))
    # Adjust for the top crop
    i -= top_crop_pixels
    # Adjust for the left crop
    j -= left_crop_pixels
    return i, j

# --- Load map data from JSON file ---
with open('map-1748449602913/data.json') as f:
    map_json = json.load(f)['map']
    map_data = map_json['data']

rows = map_data['rows']
cols = map_data['cols']
origin_x = map_json['origin_x']
origin_y = map_json['origin_y']
resolution = map_json['resolution']

# --- Extract and process map data ---
data = map_data['data']
data_base64 = map_data.get('data_base64', None)

if len(data) == 0 and data_base64 and len(data_base64) > 0:
    decoded_data = base64.b64decode(data_base64)
    with gzip.GzipFile(fileobj=BytesIO(decoded_data)) as f:
        decompressed_data = f.read()
    data = base64.b64decode(decompressed_data)
    data = np.frombuffer(data, dtype=np.uint8)
    data = data.copy()
    data[data == 255] = 0
else:
    data = np.array(data, dtype=np.uint8)
    data[data == 255] = 0

# --- Reshape data to 2D array and rotate 90 degrees CCW ---
map_array = data.reshape((rows, -1))
map_array_rot = np.rot90(map_array)

# Mirror the array across the horizontal axis
map_array_rot = np.flipud(map_array_rot)

# crop the image to the desired area
top_crop_pixels = 100
bottom_crop_pixels = 230
left_crop_pixels = 20
right_crop_pixels = 10
map_array_rot = map_array_rot[top_crop_pixels:-bottom_crop_pixels, left_crop_pixels:-right_crop_pixels]

new_cols = map_array_rot.shape[1]
new_rows = map_array_rot.shape[0]
dpi = 300
figsize = (new_cols / dpi, new_rows / dpi)

fig = plt.figure(figsize=figsize, dpi=dpi)
plt.imshow(map_array_rot, cmap='gist_yarg')

# --- Draw dots at pixel locations ---
# Origin (0, 0) in meters
i0, j0 = meters_to_rotated_pixel(0, 0, origin_x, origin_y, resolution, top_crop_pixels, left_crop_pixels)
# Debug
print(f"Origin meters: ({origin_x}, {origin_y})")
print(f"Origin pixel: ({i0}, {j0})")

plt.plot(j0, i0, 'ro', markersize=5, label='Origin [0,0] (pixel)')

# (3.371031, 24.39235) in meters
i1, j1 = meters_to_rotated_pixel(3.371031, 24.39235, origin_x, origin_y, resolution, top_crop_pixels, left_crop_pixels)
plt.plot(j1, i1, 'go', markersize=5, label='(3.37, 24.39) (pixel)')

print(f"Green dot meters: (3.371031, 24.39235)")
print(f"Green dot pixel: ({i1}, {j1})")

i2, j2 = meters_to_rotated_pixel(4.418724060058594, 15.4932222366333, origin_x, origin_y, resolution, top_crop_pixels, left_crop_pixels)
plt.plot(j2, i2, 'go', markersize=5, label='ev')
# plt.title("Map with pixel dots (rotated)")
# plt.legend()
plt.axis('off')
plt.savefig('map_plot.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.show()