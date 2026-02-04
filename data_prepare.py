import os
import xml.etree.ElementTree as ET
import shutil
from xml.dom import minidom
from PIL import Image
import random

class DataProcessor:
    def __init__(self):
        # Input paths
        self.input_anno_path = "/data/26fa1f99/dataset/DIOR_RSVG/Annotations"
        self.input_images_path = "/data/26fa1f99/dataset/DIOR_RSVG/JPEGImages"

        # Output paths
        self.output_anno_path = "/data/26fa1f99/dataset/DIOR_RSVG_p/Annotations"
        self.output_images_path = "/data/26fa1f99/dataset/DIOR_RSVG_p/JPEGImages"

        # Create output directories
        os.makedirs(self.output_anno_path, exist_ok=True)
        os.makedirs(self.output_images_path, exist_ok=True)
        os.makedirs("/data/26fa1f99/dataset/DIOR_RSVG_p", exist_ok=True)

        self.count = 0
        self.original_count = 0  # To track number of original samples

    def process_single_object(self, anno_file_path, original_image_path):
        """Process each xml file"""
        root = ET.parse(anno_file_path).getroot()
        img = Image.open(original_image_path)
        img_width, img_height = img.size

        for member in root.findall('object'):

            bndbox = member.find('bndbox')
            if bndbox is None:
                print(f"Warning: No bounding box found in {anno_file_path}")
                continue

            # Step 1: Save original split version
            new_root = ET.Element("annotation")
            filename_element = ET.SubElement(new_root, "filename")
            filename_element.text = f"{self.count:05d}.jpg"

            # Copy basic information
            source_element = ET.SubElement(new_root, "source")
            database_element = ET.SubElement(source_element, "database")
            database_element.text = "DIOR"

            # Copy size information
            size_element = ET.SubElement(new_root, "size")
            width_element = ET.SubElement(size_element, "width")
            width_element.text = str(img_width)
            height_element = ET.SubElement(size_element, "height")
            height_element.text = str(img_height)
            depth_element = ET.SubElement(size_element, "depth")
            depth_element.text = "3"

            # Copy object information with fixed order
            object_element = ET.SubElement(new_root, "object")

            # 1. name element (first)
            name_element = ET.SubElement(object_element, "name")
            name_element.text = member.find('name').text

            # 2. pose element (second)
            pose_element = ET.SubElement(object_element, "pose")
            pose_element.text = "Unspecified"

            # 3. bndbox element (third)
            bndbox_element = ET.SubElement(object_element, "bndbox")
            for coord in ['xmin', 'ymin', 'xmax', 'ymax']:
                coord_elem = ET.SubElement(bndbox_element, coord)
                coord_elem.text = member.find(f'bndbox/{coord}').text

            # 4. description element (fourth)
            description_element = ET.SubElement(object_element, "description")
            original_description = member.find('description')
            if original_description is not None:
                description_element.text = original_description.text
            else:
                description_element.text = member.find('name').text  # Fallback to name if no description

            # Save split XML and copy original image
            xml_str = ET.tostring(new_root, encoding='utf-8')
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

            with open(os.path.join(self.output_anno_path, f"{self.count:05d}.xml"), "w") as f:
                f.write(pretty_xml)

            shutil.copy(original_image_path,
                       os.path.join(self.output_images_path, f"{self.count:05d}.jpg"))

            self.count += 1

        self.original_count = self.count  # Save the count of original samples

    def process_all(self):
        """Process all files"""
        def filelist(root, file_type):

            if not file_type.startswith("."):
                file_type = f".{file_type}"
            xml_files = []

            for dirpath, _, filenames in os.walk(root):
                for f in filenames:
                    if f.lower().endswith(file_type):
                        xml_files.append(os.path.join(dirpath, f))
            return xml_files



        annotations = filelist(self.input_anno_path, '.xml')

        print(f"Searching XML files in: {self.input_anno_path}")
        print(f"Found {len(annotations)} XML files in annotations directory")
        print(f"Found {len(annotations)} XML files in annotations directory")

        # First pass: create split versions
        print("Processing original split versions...")
        for anno_file_path in annotations:
            root = ET.parse(anno_file_path).getroot()
            image_file_name = root.find("./filename").text
            print(f"XML filename: {image_file_name} | XML path: {anno_file_path}")
            original_image_path = os.path.join(self.input_images_path, image_file_name)
            print(f"Checking image: {original_image_path}")


            if os.path.exists(original_image_path):
                self.process_single_object(anno_file_path, original_image_path)
            else:
                print(f"Image file not found: {original_image_path}")

        print(f"Completed original split versions. Generated {self.original_count} samples")


        if os.path.exists("/data/26fa1f99/dataset/DIOR_RSVG/train.txt"):

            with open("/root/autodl-tmp/LQVG-main/data/DIOR_RSVG/train.txt", "r") as f:
                original_indices = [int(line.strip()) for line in f.readlines()]

            # create train.txt
            with open("/data/26fa1f99/dataset/DIOR_RSVG_p/train.txt", "w") as f:

                for idx in original_indices:
                    f.write(f"{idx}\n")


        # copy val.txt and test.txt
        for split in ["val.txt", "test.txt"]:
            src = os.path.join("/data/26fa1f99/dataset/DIOR_RSVG", split)
            dst = os.path.join("/data/26fa1f99/dataset/DIOR_RSVG_p", split)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"Copied {src} to {dst}")
            else:
                print(f"{src} does not exist, skipping.")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all()