import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import glob
import matplotlib.pyplot as plt

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array."""
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Print debug information
    print("Detection Scores:", output_dict['detection_scores'])
    print("Detection Classes:", output_dict['detection_classes'])

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, category_index, image_path):
    # Debugging category index
    print("Category Index:", category_index)

    if os.path.isdir(image_path):
        image_paths = []
        for file_extension in ('*.png', '*.jpg', '*.jpeg'):
            image_paths.extend(glob.glob(os.path.join(image_path, file_extension)))

        # Ensure outputs directory exists
        os.makedirs('outputs', exist_ok=True)

        for i, i_path in enumerate(image_paths):
            image_np = load_image_into_numpy_array(i_path)
            
            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np)
            
            # Threshold the detection scores (e.g., only show detections above 0.5)
            threshold = 0.5
            valid_indices = output_dict['detection_scores'] > threshold
            
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'][valid_indices],
                output_dict['detection_classes'][valid_indices],
                output_dict['detection_scores'][valid_indices],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=threshold
            )
            
            plt.figure(figsize=(12,8))
            plt.imshow(image_np)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"outputs/detection_output_{i}.png", bbox_inches='tight', pad_inches=0)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside image')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to image (or folder)')
    
    # Optional threshold argument
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    
    args = parser.parse_args()

    # Load detection model
    detection_model = load_model(args.model)
    
    # Create category index from labelmap
    try:
        category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)
    except Exception as e:
        print(f"Error loading labelmap: {e}")
        print("Check your labelmap file format and path.")
        exit(1)

    # Run inference
    run_inference(detection_model, category_index, args.image_path)