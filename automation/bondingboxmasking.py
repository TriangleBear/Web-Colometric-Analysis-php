import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

# Paths
xml_folder = r"D:\Programming\Urine_Test_Strips\urine\train\labelimg mask images"
image_folder = r"D:\Programming\Urine_Test_Strips\urine\train\images"
mask_folder = r"D:\Programming\Urine_Test_Strips\urine\train\mask lables"

# Ensure mask folder exists
os.makedirs(mask_folder, exist_ok=True)

def create_mask_from_xml(xml_file, image_file, mask_file):
    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image dimensions
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    # Initialize a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each object in the XML
    for obj in root.findall("object"):
        # Get bounding box coordinates
        xmin = int(obj.find("bndbox/xmin").text)
        ymin = int(obj.find("bndbox/ymin").text)
        xmax = int(obj.find("bndbox/xmax").text)
        ymax = int(obj.find("bndbox/ymax").text)

        # Draw the bounding box on the mask
        mask[ymin:ymax, xmin:xmax] = 255  # Foreground labeled as 255

    # Save the mask
    cv2.imwrite(mask_file, mask)
    print(f"Saved mask: {mask_file}")

# Process all XML files
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(xml_folder, xml_file)

        # Get corresponding image and mask filenames
        base_name = os.path.splitext(xml_file)[0]
        image_file = os.path.join(image_folder, base_name + ".jpg")
        mask_file = os.path.join(mask_folder, base_name + ".png")

        if os.path.exists(image_file):
            create_mask_from_xml(xml_path, image_file, mask_file)
        else:
            print(f"Image not found for: {xml_file}")

print("Mask generation completed!")
