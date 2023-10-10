import shutil
import matplotlib.pyplot as plt
import glob
import cv2
import os
import random
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Previous class -> new class
# Key -> Value
class_id_mapping = {
    '0': '9',
    '1': '9',
    '2': '9',
    '3': '9',
    '4': '9',
    '5': '9',
    '6': '9',
    '8': '9',
    '9': '9',
    '7': '8'
}

def create_directories_if_not_exist(directory):
    os.makedirs(directory, exist_ok=True)

def yolo2tobox(bboxes):
    xmin, ymin = bboxes[0] - bboxes[2]/2, bboxes[1] - bboxes[3]/2
    xmax, ymax = bboxes[0] + bboxes[2]/2, bboxes[1] + bboxes[3]/2

    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, label):
    h, w, _ = image.shape
    for idx, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2tobox(box)
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)

        thickness = max(2, int(w/275))

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=thickness
        )
        cv2.putText(
            image, 
            str(label[idx][0]),
            (xmin, ymin),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 25),
            2
        )

    return image

def process_images(source_dir:str, target_dir:str, is_modify_class:bool = False):
    total_images = 0
    total_labels = 0

    logging.info(f"Working on {os.path.basename(source_dir)} data.....")
    for dataset in ['train', 'test', 'valid']:
        source_dataset_dir = os.path.join(source_dir, dataset)
        target_dataset_dir = os.path.join(target_dir, dataset)
        create_directories_if_not_exist(target_dataset_dir)

        for data_type in ['images', 'labels']:
            source_data_dir = os.path.join(source_dataset_dir, data_type)
            target_data_dir = os.path.join(target_dataset_dir, data_type)
            create_directories_if_not_exist(target_dataset_dir)

            image_paths = glob.glob(os.path.join(source_data_dir, '*.jpg'))            
            label_paths = glob.glob(os.path.join(source_data_dir.replace('images', 'labels'), '*.txt'))

            logger.info(f"{dataset.upper()} Dataset - {data_type.capitalize()}:")
            logger.info(f"  Images: {len(image_paths)}")
            logger.info(f"  Labels: {len(label_paths)}")

            if data_type == 'images':
                total_images += len(image_paths)
            elif data_type == 'labels':
                total_labels += len(label_paths)

            for image_path in image_paths:
                base_filename = os.path.splitext(os.path.basename(image_path))[0]

                # Load corresponding label file
                label_file_path = os.path.join(source_data_dir.replace('images', 'labels'), base_filename + '.txt')
                if not os.path.exists(label_file_path):
                    continue

                with open(label_file_path, 'r') as label_file:
                    modified_data = []
                    bboxes = []
                    labels = []
                    label_lines = label_file.readlines()

                    for line in label_lines:
                        parts = line.strip().split(' ')
                        
                        # Check if there are at least five values in the parts
                        if len(parts) < 6:
                            class_id = parts[0]
                            bbox_str = ' '.join(parts[1:])
                            
                            x_ctr, y_ctr, w, h = bbox_str.split(' ')
                            x_ctrf = float(x_ctr)
                            y_ctrf = float(y_ctr)
                            wf = float(w)
                            hf = float(h)

                            if not is_modify_class:
                                bboxes.append([x_ctrf, y_ctrf, wf, hf])
                                labels.append([class_id])
                                modified_data.append((class_id, x_ctr, y_ctr, w, h))  # Append class_id here
                            else:
                                label = ""
                                if class_id in class_id_mapping:
                                    label = class_id_mapping[class_id]
                                    labels.append([label])
                                    bboxes.append([x_ctrf, y_ctrf, wf, hf])
                                
                                modified_data.append((label, x_ctr, y_ctr, w, h))
                        else:
                            # Print information about the skipped line
                            file_name = os.path.basename(label_file.name)
                            num_objects = len(label_lines)
                            logger.warning(f"Skipping line in {file_name}: class_id={class_id}, num_objects={num_objects}")
                            logger.warning(f"Line content: {line.strip()}")
                        
                new_label_file_path = os.path.join(target_data_dir.replace('images', 'labels'), base_filename + '.txt')
                os.makedirs(os.path.dirname(new_label_file_path), exist_ok=True)

                if os.path.exists(new_label_file_path):
                    # If it exists, open it in append mode
                    with open(new_label_file_path, 'a') as new_label_file:
                        for data in modified_data:
                            label, x_ctr, y_ctr, w, h = data
                            new_label_file.write(f"{label} {x_ctr} {y_ctr} {w} {h}\n")
                else:
                    # If it doesn't exist, create a new label file and write to it
                    with open(new_label_file_path, 'w') as new_label_file:
                        for data in modified_data:
                            label, x_ctr, y_ctr, w, h = data
                            new_label_file.write(f"{label} {x_ctr} {y_ctr} {w} {h}\n")
                
                os.makedirs(target_data_dir, exist_ok=True)
                new_image_path = os.path.join(target_data_dir, os.path.basename(image_path))
                shutil.copy(image_path, new_image_path)

    logger.info("\n")
    logger.info(f"  Total -> Images: {total_images} processed")
    logger.info(f"  Total -> Labels: {total_labels} processed")


def show_detections(image_paths, label_paths):
    num_samples = 4
    all_images = []
    all_images.extend(glob.glob(os.path.join(image_paths, '*.jpg')))
    all_images.extend(glob.glob(os.path.join(image_paths, '*.jpeg')))
    all_images.extend(glob.glob(os.path.join(image_paths, '*.JPG')))

    # Shuffle the list of images only once to avoid duplicates
    random.shuffle(all_images)

    class_label = {
        '0': 'bicycle',
        '1': 'bus',
        '2': 'crosswalk',
        '3': 'fire hydrant',
        '4': 'motorcycle',
        '5': 'traffic light',
        '6': 'vehicle',
        '7': 'truck',
        '8': 'person',
        '9': 'animal'
    }

    plt.figure(figsize=(15, 12))
    num_rows = 2
    num_cols = 2
    idx = 0

    for i in range(num_rows):
        for j in range(num_cols):
            if idx >= num_samples:
                break

            j = random.randint(0, len(all_images) - 1)
            image_name = all_images[j]
            image_name = '.'.join(image_name.split(os.path.sep)[-1].split('.')[:-1])

            image = cv2.imread(all_images[j])
            with open(os.path.join(label_paths, image_name + '.txt'), 'r') as f:
                bboxes = []
                labels = []
                label_lines = f.readlines()

            for line in label_lines:
                class_id = line[0]
                bbox_str = line[2:]
                x_ctr, y_ctr, w, h = bbox_str.split(' ')
                x_ctrf = float(x_ctr)
                y_ctrf = float(y_ctr)
                wf = float(w)
                hf = float(h)

                if class_id in class_label:
                    label = class_label[class_id]

                bboxes.append([x_ctrf, y_ctrf, wf, hf])
                labels.append([label])

            result_img = plot_box(image, bboxes, labels)
            plt.subplot(num_rows, num_cols, idx + 1)
            plt.imshow(result_img[:, :, ::-1])
            plt.axis('off')
            idx += 1

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.show()



def main(args):
    if args.mode == 'show-detections':
        show_detections(
        os.path.join(args.target_dir, 'train', 'images'),
        os.path.join(args.target_dir, 'train', 'labels'),
    )
    if args.mode == 'prepare-detections':
        process_images(args.source_dir, args.target_dir, args.is_modify_class)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and labels.")
    parser.add_argument("--source-dir", type=str, required=False, help="Source directory containing the images and labels.")
    parser.add_argument("--target-dir", type=str, required=True, help="Target directory to save processed images and labels.")
    parser.add_argument('--mode', type=str, choices=['show-detections', 'prepare-detections'], default='prepare detections',
                        help="Choose 'show detections' to display detections or 'prepare detections' to process images and labels.")
    parser.add_argument('--is-modify-class', action='store_true', default=False,
                        help="Specify whether to modify class_id with class_id_mapping.")
    args = parser.parse_args()
    main(args)