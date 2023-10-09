import shutil
import matplotlib.pyplot as plt
import glob
import cv2
import os
import random

# Define your class_id_mapping
class_id_mapping = {
    '0': '1',
    '2': '1',
    '3': '1',
    '6': '1',
    '1': '7',
    '5': '7',
    '7': '7',
    '8': '7',
    '9': '7',
    '10': '7',
    '11': '7',
    '4': '6',
}

source_directory = os.path.join(os.getcwd(),"vehicles.v1")
target_directory = os.path.join(os.getcwd(),'combined')
os.makedirs(target_directory, exist_ok=True)

def create_directories_if_not_exist(directory):
    os.makedirs(directory, exist_ok=True)


def process_images(source_dir, target_dir):
    total_images = 0
    total_labels = 0

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

            print(f"{dataset.upper()} Dataset - {data_type.capitalize()}:")
            print(f"  Images: {len(image_paths)}")
            print(f"  Labels: {len(label_paths)}")

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
                        
                        # Check if there are at least four values in the line
                        if len(parts) >= 5:
                            class_id = parts[0]
                            bbox_str = ' '.join(parts[1:])
                            
                            x_ctr, y_ctr, w, h = bbox_str.split(' ')
                            x_ctrf = float(x_ctr)
                            y_ctrf = float(y_ctr)
                            wf = float(w)
                            hf = float(h)

                            if class_id in class_id_mapping:
                                label = class_id_mapping[class_id]

                            bboxes.append([x_ctrf, y_ctrf, wf, hf])
                            labels.append([label])

                            modified_data.append((label, x_ctr, y_ctr, w, h))
                        else:
                            # Print information about the skipped line
                            file_name = os.path.basename(label_file.name)
                            num_objects = len(label_lines)
                            print(f"Skipping line in {file_name}: class_id={class_id}, num_objects={num_objects}")
                            print(f"Line content: {line.strip()}")
                        
                new_label_file_path = os.path.join(target_data_dir.replace('images', 'labels'), base_filename + '.txt')
                os.makedirs(os.path.dirname(new_label_file_path), exist_ok=True)

                with open(new_label_file_path, 'w') as new_label_file:
                    for data in modified_data:
                        label, x_ctr, y_ctr, w, h = data
                        new_label_file.write(f"{label} {x_ctr} {y_ctr} {w} {h}\n")
                
                os.makedirs(target_data_dir, exist_ok=True)
                new_image_path = os.path.join(target_data_dir, os.path.basename(image_path))
                shutil.copy(image_path, new_image_path)

    print("\nTotal ->")
    print(f"  Images: {total_images} processed")
    print(f"  Labels: {total_labels} processed")



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
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 25),
            2
        )

    return image

def show_detections(image_paths, label_paths, num_samples:int = 4):

    all_images = []
    all_images.extend(glob.glob(os.path.join(image_paths, '*.jpg')))
    all_images.extend(glob.glob(os.path.join(image_paths, '*.jpeg')))
    all_images.extend(glob.glob(os.path.join(image_paths, '*.JPG')))

    all_images.sort()

    class_label = {
    '1':'bus',
    '1':'bus',
    '1':'bus',
    '1':'bus',
    '7':'truck',
    '7':'truck',
    '7':'truck',
    '7':'truck',
    '7':'truck',
    '7':'truck',
    '7':'truck',
    '6':'vehicle'
    }

    # num_samples = len(all_images)
    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0,num_samples-1)
        image_name = all_images[j]
        image_name = '.'.join(image_name.split(os.path.sep)[-1].split('.')[:-1])
        image = cv2.imread(all_images[j])
        with open(os.path.join(label_paths, image_name+'.txt'), 'r') as f:
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
        
        
        result_img =  plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_img[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=1)
    plt.tight_layout()
    plt.show()

# process_images(source_directory, target_directory)

show_detections(
    os.path.join(os.getcwd(),'combined','train', 'images'),
    os.path.join(os.getcwd(),'combined','train', 'labels'),
    4
)
