import os

# Directories to compare
images_dir = './yolo_dataset/SPA/images/val'
labels_dir = './yolo_dataset/SPA/labels/val'

# Get filenames without extensions
images_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir)}
labels_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)}

# Find differences
only_in_images = images_files - labels_files
only_in_labels = labels_files - images_files

# Print results
print("Files only in images directory:")
for file in only_in_images:
    print(file)

print("\nFiles only in labels directory:")
for file in only_in_labels:
    print(file)