import os

def delete_empty_annotation_files_with_images(labels_folder, images_folder, image_extensions=("jpg", "png")):
    """
    Delete empty annotation files and their corresponding image files.

    Args:
        labels_folder (str): Path to the folder containing annotation files.
        images_folder (str): Path to the folder containing image files.
        image_extensions (tuple): Tuple of possible image file extensions (default is ("jpg", "png")).
    """
    # Counter for deleted files
    deleted_files_count = 0

    # Iterate through files in the labels folder
    for label_file in os.listdir(labels_folder):
        label_path = os.path.join(labels_folder, label_file)

        # Check if the annotation file is empty
        if os.path.isfile(label_path) and os.path.getsize(label_path) == 0:
            # Delete the empty annotation file
            os.remove(label_path)
            print(f"Deleted empty annotation file: {label_path}")
            deleted_files_count += 1

            # Check for corresponding image file and delete it
            base_name = os.path.splitext(label_file)[0]
            for ext in image_extensions:
                image_path = os.path.join(images_folder, f"{base_name}.{ext}")
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Deleted corresponding image file: {image_path}")
                    break

    print(f"Total empty annotation files deleted: {deleted_files_count}")

if __name__ == "__main__":
    labels_folder_path = "../datasets/urine/train/labels"
    images_folder_path = "../datasets/urine/train/images"
    delete_empty_annotation_files_with_images(labels_folder_path, images_folder_path)
