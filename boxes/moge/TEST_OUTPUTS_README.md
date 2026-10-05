# Test Outputs

## Location

All inference outputs are saved to: `images/moge_box/test_output/`

Each test run creates a timestamped directory:
```
test_output/
├── basic_inference_20240915_103045/
│   ├── depth_map.png           # Colorized depth visualization
│   ├── depth_raw.npy           # Raw depth values (numpy array)
│   ├── normal_map.png          # Surface normals (RGB: xyz)
│   ├── mask.png                # Valid pixel mask (white=valid)
│   ├── input_image.jpg         # Original input image
│   └── pointcloud.csv          # 3D points with colors (XYZRGB)
├── batch_inference_20240915_104215/
│   └── ...
```

## File Formats

### depth_map.png
**Format**: JET colormap [0-255]  
**Interpretation**: 
- Blue = Close objects (smaller depth values)
- Red = Far objects (larger depth values)
- White/Black = Invalid pixels

**To view**: Any image viewer, Photoshop, GIMP

### depth_raw.npy
**Format**: NumPy float32 array  
**Shape**: (H, W)  
**Units**: Meters  
**To load**:
```python
import numpy as np
depth = np.load('depth_raw.npy')
print(f"Depth range: {depth.min():.2f}m - {depth.max():.2f}m")
```

### normal_map.png
**Format**: RGB image [0-255]  
**Interpretation**: Normal vectors encoded as RGB colors
- R = X component (right)
- G = Y component (down)
- B = Z component (forward)
- Values: [-1, 1] → [0, 255]

**To view**: Image viewer, or decode:
```python
normal = (normal_map / 255 * 2 - 1)  # back to [-1, 1]
```

### mask.png
**Format**: Grayscale binary mask [0, 255]  
**Interpretation**:
- White (255) = Valid pixel (geometry estimated)
- Black (0) = Invalid pixel (no prediction)

**Use**: Filter depth/point cloud to valid pixels only

### pointcloud.csv
**Format**: CSV with headers  
**Columns**: x, y, z, r, g, b  
**Shape**: N valid points  
**Units**: Meters (camera coordinates)  
**To view**: MeshLab, Blender, CloudCompare

**Import in MeshLab**:
1. File → Import Mesh
2. Select CSV file
3. Enable "ASCII format" if needed

## Viewing Instructions

### Quick View (depth)
```bash
# Open depth map in default image viewer
xdg-open test_output/basic_inference_*/depth_map.png

# View on Linux
eog test_output/basic_inference_*/depth_map.png

# View on macOS
open test_output/basic_inference_*/depth_map.png
```

### Python Viewing Script
```python
import matplotlib.pyplot as plt
import numpy as np

# Load depth
depth = np.load('test_output/basic_inference_*/depth_raw.npy')
mask = cv2.imread('test_output/basic_inference_*/mask.png', 0)

# Show depth and mask
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Depth
depth_vis = depth.copy()
depth_vis[mask > 0] = (depth_vis[mask > 0] - depth[mask>0].min()) / \
                      (depth[mask>0].max() - depth[mask>0].min() + 1e-6)
axes[0].imshow(depth_vis, cmap='gray')
axes[0].set_title(f'Depth: {depth[mask>0].min():.1f}m - {depth[mask>0].max():.1f}m')
axes[0].axis('off')

# Mask
axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Valid Mask')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### Point Cloud Visualization
```python
import pandas as pd
import open3d as o3d

# Load CSV
df = pd.read_csv('test_output/basic_inference_*/pointcloud.csv')
points = df[['x', 'y', 'z']].values
colors = df[['r', 'g', 'b']].values / 255.0

# Create mesh
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])
```

## Troubleshooting

### Depth map looks all white or all black
- The mask is probably all black (no valid pixels)
- Try a different input image with more features
- MoGe works best with textured surfaces

### Point cloud looks distorted
- Camera FOV not provided (default FOV used)
- Add `fov_x` parameter: `{"moge": {"parameters": {"fov_x": 60.0}}}`
- FOV is typically 45-90 degrees for most cameras

### Normal map is mostly gray
- Model may be predicting flat surface normals
- This can happen with very uniform images
- Try with a different input image

### Point cloud missing colors
- Input image wasn't successfully copied
- Check that input_image.jpg exists in the output directory
- Color sampling is approximate (uses simple image coordinates)

## Next Steps

Once you've reviewed the outputs and they look good:

1. **Adjust parameters**: Try different `refine_steps`, `resolution_level` values
2. **Add more test images**: Your own photos for real-world testing
3. **Compare with other boxes**: Use `vggt` for stereo depth, `clip` for semantics
4. **Integration**: Start building workflows with multiple boxes

Enjoy exploring 3D geometry from single images! 🎨📸
