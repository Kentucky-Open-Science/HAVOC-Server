# import json
# import base64
# import gzip
# from PIL import Image
# import os
# import numpy as np  # Import numpy for potential data type handling
#
#
# def save_map_image_from_json(json_file_path, output_image_path="map_image.png"):
#     """
#     Loads map data from a JSON file, decodes and decompresses the map image,
#     and saves it as a PNG file using derived dimensions.
#     """
#     try:
#         print(f"Attempting to open and load JSON file: {json_file_path}")
#         with open(json_file_path, 'r') as f:
#             data = json.load(f)
#         print("JSON file loaded successfully.")
#
#         # Extract necessary fields from JSON
#         base64_encoded_gzipped_data = data.get('map', {}).get('data', {}).get('data_base64')
#         if not base64_encoded_gzipped_data:
#             print("Error: 'data_base64' field not found.")
#             return
#
#         # Get the stated columns (width) from the JSON
#         # This will be used to determine the true rows of the data buffer
#         actual_cols = data.get('map', {}).get('data', {}).get('cols')
#         if actual_cols is None:
#             actual_cols = data.get('map', {}).get('width')  # Fallback to map.width
#
#         if actual_cols is None:
#             print("Error: Map columns/width not found in JSON.")
#             return
#
#         # Get the logical/ROI height if needed for cropping later
#         logical_rows_roi = data.get('map', {}).get('data', {}).get('rows')
#         if logical_rows_roi is None:
#             logical_rows_roi = data.get('map', {}).get('height')  # Fallback
#
#         print(f"Stated columns (width) from JSON: {actual_cols}")
#         if logical_rows_roi:
#             print(f"Logical rows (height) from JSON (for ROI): {logical_rows_roi}")
#
#         print("Decoding base64 string...")
#         gzipped_data = base64.b64decode(base64_encoded_gzipped_data)
#         print("Base64 decoding successful.")
#
#         print("Decompressing gzipped data...")
#         raw_image_data_bytes = gzip.decompress(gzipped_data)
#         print(f"Gzip decompression successful. Total data length: {len(raw_image_data_bytes)} bytes.")
#
#         # Determine the actual number of rows in the data buffer
#         if len(raw_image_data_bytes) % actual_cols != 0:
#             print(f"Warning: Total data length ({len(raw_image_data_bytes)}) "
#                   f"is not perfectly divisible by the number of columns ({actual_cols}).")
#             # You might want to handle this case, e.g., by truncating or erroring
#             # For now, we'll use integer division, which effectively truncates.
#         actual_rows = len(raw_image_data_bytes) // actual_cols
#
#         print(f"Derived actual matrix dimensions: Width={actual_cols}, Height={actual_rows}")
#
#         # Check if the derived dimensions make sense for the data length
#         expected_data_length = actual_rows * actual_cols
#         if len(raw_image_data_bytes) != expected_data_length:
#             print(f"Error: Derived data length ({expected_data_length}) "
#                   f"does not match actual decompressed data length ({len(raw_image_data_bytes)}).")
#             print("This suggests an issue with the data or column count. The remainder of the division was non-zero.")
#             # This should ideally not happen if the previous modulo check was handled or passed.
#             # If there was a remainder, raw_image_data_bytes might need truncation:
#             # raw_image_data_bytes = raw_image_data_bytes[:expected_data_length]
#
#         # dt: "c" suggests 8-bit data. Common for maps is unsigned (0-255).
#         # Pillow 'L' mode is for 8-bit grayscale.
#         # If dt:"c" implied signed char (CV_8S, -128 to 127), you might need to convert
#         # the data to unsigned or use a different approach if Pillow struggles.
#         # For now, assume it can be interpreted as unsigned 8-bit for 'L' mode.
#         # raw_image_data_np = np.frombuffer(raw_image_data_bytes, dtype=np.uint8) # Or np.int8 if 'c' is signed
#         # if dt is signed, you might convert: raw_image_data_np = (raw_image_data_np + 128).astype(np.uint8)
#         # image_data_for_pillow = raw_image_data_np.tobytes()
#
#         print("Creating image from raw data using derived dimensions...")
#         image = Image.frombytes('L', (actual_cols, actual_rows), raw_image_data_bytes)
#         print("Full image object (based on derived dimensions) created.")
#
#         # Optional: Crop to the logical ROI if specified and different
#         if logical_rows_roi and logical_rows_roi < actual_rows:
#             print(f"Cropping image from {actual_cols}x{actual_rows} to {actual_cols}x{logical_rows_roi}")
#             # Box is (left, upper, right, lower)
#             # We assume the ROI starts from the top (0,0) of the larger image
#             image = image.crop((0, 0, actual_cols, logical_rows_roi))
#             print("Image cropped to logical ROI.")
#         elif logical_rows_roi and logical_rows_roi > actual_rows:
#             print(
#                 f"Warning: Logical rows ROI ({logical_rows_roi}) is greater than actual derived rows ({actual_rows}). Using full derived image.")
#
#         print(f"Saving image to {output_image_path}...")
#         image.save(output_image_path)
#         print(f"Image saved successfully as {output_image_path}")
#
#     except FileNotFoundError:
#         print(f"Error: JSON file not found at {json_file_path}")
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from {json_file_path}. Check if it's a valid JSON.")
#     except base64.binascii.Error as e:
#         print(f"Error: Base64 decoding failed. {e}")
#     except gzip.BadGzipFile as e:
#         print(f"Error: Could not decompress data. It might not be a valid gzip file. {e}")
#     except ValueError as e:
#         print(f"Error creating image from bytes. This might be due to incorrect data length or format: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#
#
# if __name__ == '__main__':
#     json_file_to_process = "data.json"  # Replace with your JSON file path
#     output_image_file = "map_image_derived.png"  # Changed output name
#
#     if not os.path.exists(json_file_to_process):
#         print(f"Error: The input file '{json_file_to_process}' was not found.")
#         print("Please make sure it's in the same directory as this script, or update the path.")
#     else:
#         print("Found file, processing...")
#         save_map_image_from_json(json_file_to_process, output_image_file)

import json
import numpy as np
import matplotlib.pyplot as plt
import base64
import gzip
from io import BytesIO

with open("data.json") as f:
  map = json.load(f)['map']
  js = map['data']

print(js['rows'])

print(js['cols'])
print(map['origin_x'])
print(-map['origin_x']/map['resolution'])

data = js['data']
data_base64 = js.get('data_base64', None)
print(len(data))

if (len(data) == 0 and len(data_base64) > 0):
  # Parse data_base64 instead

  # Decode the base64 data
  decoded_data = base64.b64decode(data_base64)

  # Decompress the gzipped data
  with gzip.GzipFile(fileobj=BytesIO(decoded_data)) as f:
    decompressed_data = f.read()
  data = base64.b64decode(decompressed_data)

  data = np.frombuffer(data, dtype=np.uint8)
  data = data.copy()
  data[data == 255] = 0

  print(len(data))
  print(data[:100])


a = np.array(data)
nslices = js['rows']
b = a.reshape((nslices, -1))

# This is the flipped real map

print(js['cols']+map['origin_x']/map['resolution'])

c = plt.imshow(np.flip(b, -1), cmap='gist_yarg')
plt.colorbar(c)

# draw [0, 0] point
plt.plot(js['cols']+map['origin_x']/map['resolution'], -map['origin_y']/map['resolution'], 'ro')


plt.title(f"Map size: {round(js['rows']*map['resolution'], 2)}x{round(js['cols']*map['resolution'], 2)}")

plt.show()