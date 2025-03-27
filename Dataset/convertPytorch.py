import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import pandas as pd
import glob


def yolo_to_labelimg(yolo_dir, image_dir, output_dir, classes, base_path=None):
    """
    Convert YOLO v5 annotation format to LabelImg XML format.

    Args:
    - yolo_dir: Directory containing YOLO .txt annotation files
    - image_dir: Directory containing corresponding image files
    - output_dir: Directory to save XML annotation files
    - classes: List of class names in the same order as YOLO class indices
    - base_path: Optional base path for the XML 'path' element
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through YOLO annotation files
    for yolo_file in os.listdir(yolo_dir):
        if not yolo_file.endswith('.txt'):
            continue

        # Construct file paths
        yolo_path = os.path.join(yolo_dir, yolo_file)

        # Try multiple image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
        image_path = None
        for ext in image_extensions:
            image_filename = yolo_file.replace('.txt', ext)
            potential_image_path = os.path.join(image_dir, image_filename)
            if os.path.exists(potential_image_path):
                image_path = potential_image_path
                break

        # Skip if no corresponding image is found
        if not image_path:
            print(f"Warning: No image found for {yolo_file}")
            continue

        # Create XML annotation
        annotation = create_xml_annotation(image_path, yolo_path, classes, base_path)

        # Save XML file
        xml_filename = yolo_file.replace('.txt', '.xml')
        xml_path = os.path.join(output_dir, xml_filename)

        # Pretty print and save XML
        xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
        with open(xml_path, 'w') as f:
            f.write(xml_str)

        print(f"Converted {yolo_file} to {xml_filename}")


def create_xml_annotation(image_path, yolo_path, classes, base_path=None):
    """
    Create XML annotation structure from YOLO format.

    Args:
    - image_path: Full path to the image file
    - yolo_path: Full path to the YOLO annotation file
    - classes: List of class names
    - base_path: Optional base path for the XML 'path' element

    Returns:
    - XML ElementTree annotation
    """
    from PIL import Image

    # Get image dimensions
    img = Image.open(image_path)
    width, height = img.size

    # Create root annotation element
    annotation = ET.Element('annotation')

    # Add folder
    folder_elem = ET.SubElement(annotation, 'folder')
    folder_elem.text = os.path.basename(os.path.dirname(image_path))

    # Add filename
    filename_elem = ET.SubElement(annotation, 'filename')
    filename_elem.text = os.path.basename(image_path)

    # Add path
    path_elem = ET.SubElement(annotation, 'path')
    path_elem.text = base_path or image_path

    # Add source
    source_elem = ET.SubElement(annotation, 'source')
    database_elem = ET.SubElement(source_elem, 'database')
    database_elem.text = 'Unknown'

    # Add size information
    size_elem = ET.SubElement(annotation, 'size')
    width_elem = ET.SubElement(size_elem, 'width')
    width_elem.text = str(width)
    height_elem = ET.SubElement(size_elem, 'height')
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size_elem, 'depth')
    depth_elem.text = str(3)  # Assuming 3-channel image

    # Add segmented
    segmented_elem = ET.SubElement(annotation, 'segmented')
    segmented_elem.text = '0'

    # Read YOLO annotations
    with open(yolo_path, 'r') as f:
        for line in f:
            # Parse YOLO format: <class_id> <x_center> <y_center> <width> <height>
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Warning: Invalid YOLO annotation in {yolo_path}")
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width_normalized = float(parts[3])
            height_normalized = float(parts[4])

            # Convert YOLO normalized coordinates to pixel coordinates
            x_min = int((x_center - width_normalized / 2) * width)
            y_min = int((y_center - height_normalized / 2) * height)
            x_max = int((x_center + width_normalized / 2) * width)
            y_max = int((y_center + height_normalized / 2) * height)

            # Ensure coordinates are within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            # Create object element
            object_elem = ET.SubElement(annotation, 'object')
            name_elem = ET.SubElement(object_elem, 'name')

            # Ensure class_id is within the classes list range
            if 0 <= class_id < len(classes):
                name_elem.text = classes[class_id]
            else:
                name_elem.text = f'unknown_class_{class_id}'

            # Add additional object details
            pose_elem = ET.SubElement(object_elem, 'pose')
            pose_elem.text = 'Unspecified'
            truncated_elem = ET.SubElement(object_elem, 'truncated')
            truncated_elem.text = '0'
            difficult_elem = ET.SubElement(object_elem, 'difficult')
            difficult_elem.text = '0'

            bndbox_elem = ET.SubElement(object_elem, 'bndbox')
            xmin_elem = ET.SubElement(bndbox_elem, 'xmin')
            xmin_elem.text = str(x_min)
            ymin_elem = ET.SubElement(bndbox_elem, 'ymin')
            ymin_elem.text = str(y_min)
            xmax_elem = ET.SubElement(bndbox_elem, 'xmax')
            xmax_elem.text = str(x_max)
            ymax_elem = ET.SubElement(bndbox_elem, 'ymax')
            ymax_elem.text = str(y_max)

    return annotation


def xml_to_csv(path):
    """
    Convert XML annotations to a CSV file.

    Args:
    - path: Directory containing XML files

    Returns:
    - pandas DataFrame with annotation information
    """
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member.find('name').text,
                     int(member.find('bndbox')[0].text),
                     int(member.find('bndbox')[1].text),
                     int(member.find('bndbox')[2].text),
                     int(member.find('bndbox')[3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # Define your class names in the same order as YOLO class indices
    classes = ['Aircraft Carrier', 'Bulkers', 'Car Carrier', 'Container Ship', 'Cruise', 'DDG', 'Recreational',
               'Sailboat', 'Submarine', 'Tug']

    # Set your directories
    yolo_annotations_dir = 'test/labels'
    images_dir = 'test/images'
    output_xml_dir = 'test/xml'

    # Optional: Specify base path for XML 'path' element
    base_path = r'C:\Users\tobia\source\repos\H5\tf2\models\research\object_detection\images\test'

    # Convert YOLO annotations to XML
    yolo_to_labelimg(yolo_annotations_dir, images_dir, output_xml_dir, classes, base_path)

    # Convert XML to CSV for both train and test sets
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), (f'{folder}/xml'))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(f'{folder}_labels.csv', index=None)
        print(f'Successfully converted {folder} XML to CSV.')


if __name__ == "__main__":
    main()