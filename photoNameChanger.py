import os

# Define the root directory
root_dir = '/home/lalit/campusVision/MergeDataset'

# Walk through the directory
for foldername, subfolders, filenames in os.walk(root_dir):
    # Skip if current folder is the root folder
    if foldername == root_dir:
        continue
        
    photo_count = 1 
    
    for filename in filenames:
        # Check if the file is a photo (common image extensions)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Create new name based on folder name and counter
            new_name = f'{os.path.basename(foldername)}_{photo_count}{os.path.splitext(filename)[1]}'
            # Construct full file paths
            old_file = os.path.join(foldername, filename)
            new_file = os.path.join(foldername, new_name)
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} to {new_file}')
            photo_count += 1  # Increment the counter
