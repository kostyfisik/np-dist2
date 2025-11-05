import matplotlib.pyplot as plt
from np_dist2.grid_generator import generate_fcc_lattice, generate_sphere_points


# Generate points
N = 5000  # Number of points
# points = generate_sphere_points(N)

size = 2
points = generate_fcc_lattice(size)

print(len(points))
# Create visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot points
ax.scatter(points[0], points[1], points[2], c="blue", alpha=0.6)

# # Set equal aspect ratio
# ax.set_box_aspect([size, size, size])

# Add labels
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title(f"{N} Random Points on Unit Sphere Surface")
plt.show()

plt.savefig("sphere_points.png")
print(f"Generated array shape: {points.shape}")
