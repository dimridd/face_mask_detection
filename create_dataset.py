"""
change the annotations and images path accordingly in the FaceMaskDataset constructor

# create the face-mask dataset
python3 create_dataset.py -f 1

# display the image with bounding box detail
python3 create_dataset.py -f 0 -i './datasets/images/maksssksksss206.png' --bbox 104 128 156 204
"""
import xml.etree.ElementTree as ET
import os
from PIL import Image, ImageDraw
from torchvision import transforms
import torch
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv
import argparse


class FaceMaskDataset:
    def __init__(self, annotation_path="./datasets/annotations", image_path="./datasets/images"):

        self.annotation_files = os.listdir(annotation_path)
        self.annotation_files.sort()
        self.annotation_files_path = []
        for file in self.annotation_files:
            self.annotation_files_path.append(os.path.join(annotation_path, file))

        self.image_files = os.listdir(image_path)
        self.image_files.sort()
        self.image_files_path = []
        for file in self.image_files:
            self.image_files_path.append(os.path.join(image_path, file))

        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):

        annotation_file = self.annotation_files_path[idx]
        image_file = self.image_files_path[idx]
        image = Image.open(image_file)

        tree = ET.parse(annotation_file)
        root = tree.getroot()

        bndboxs = []
        labels = []

        for obj in root.findall("object"):
            bndbox_xml = obj.find("bndbox")
            label = obj.find("name").text
            if label == "mask_weared_incorrect":
                label = "without_mask"

            xmin = int(bndbox_xml.find("xmin").text)
            ymin = int(bndbox_xml.find("ymin").text)
            xmax = int(bndbox_xml.find("xmax").text)
            ymax = int(bndbox_xml.find("ymax").text)

            w = xmax - xmin
            h = ymax - ymin
            x = int(xmin + w / 2)
            y = int(ymin + h / 2)

            x /= image.size[0]
            w /= image.size[0]
            y /= image.size[1]
            h /= image.size[1]

            bndbox = (x, y, w, h)

            if self.transform:
                img = self.transform(image)

            bndboxs.append(torch.tensor(bndbox))
            labels.append(label)

        return labels, image_file, img, bndboxs

    def __len__(self):
        return len(self.annotation_files_path)


class CreateCSV:
    def __init__(self):
        self.ans = []
        self.face_images = FaceMaskDataset()
        self.sample_size = len(self.face_images)

    def unpack_bndbox(self, bndbox, image):
        x, y, w, h = tuple(bndbox)
        x *= image.size[0]
        w *= image.size[0]
        y *= image.size[1]
        h *= image.size[1]

        xmin = x - w / 2
        xmax = x + w / 2
        ymin = y - h / 2
        ymax = y + h / 2

        bndbox = [xmin, ymin, xmax, ymax]
        return bndbox

    def rectangle_image(self, face_images, i):

        labels, image_file, image, bndboxs = face_images[i]

        labelencoder = LabelEncoder()
        labels = labelencoder.fit_transform(labels)

        # img = transforms.ToPILImage()(image)
        # img = transforms.Resize((512, 512))(img)
        # draw = ImageDraw.Draw(img)

        img = transforms.ToPILImage()(image)
        img_pil = transforms.Resize((512, 512))(img)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_BGR2RGB)

        idx = 0

        bounding_boxes = []

        for bndbox in bndboxs:
            if labels[idx] == 1:
                text = "{}".format("no_mask")
            else:
                text = "{}".format("mask")

            xmin, ymin, xmax, ymax = self.unpack_bndbox(bndbox, img_pil)
            xmin, ymin, xmax, ymax = np.int(xmin), np.int(ymin), np.int(xmax), np.int(ymax)

            y = ymin - 10 if ymin - 10 > 10 else ymin + 10
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          (0, 100, 0), 2)
            cv2.putText(img, text, (xmin, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 2)
            bounding_boxes.append([xmin, ymin, xmax, ymax])
            idx += 1
        return img, image_file, bounding_boxes, labels

    def generate_csv(self, sample_size=None):
        for i in range(0, sample_size):
            try:
                img, image_file, bounding_boxes, labels = self.rectangle_image(self.face_images, i)
            except:
                continue
            info = []

            print("{} is loaded".format(image_file))
            for idx in range(len(bounding_boxes)):
                info.append(image_file)
                info.extend(bounding_boxes[idx])
                info.extend([labels[idx]])
                self.ans.append(info)
                info = []

        data_file = open('datasets.csv', 'w')

        header = ["image name", "xmin", "xmax", "ymin", "ymax", "labels"]
        final = [header]
        final.extend(self.ans)

        with data_file:
            # create the csv writer object
            csv_writer = csv.writer(data_file)

            csv_writer.writerows(final)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--flag', required=True, help='create/display the dataset on value of flag')
    ap.add_argument('-i', '--image', required=False, help='path to face image file')
    ap.add_argument('--bbox', nargs='+', required=False, help='add bounding box info in "xmin, ymin, xmax, ymax" '
                                                              'format when flag is set to 0',)
    args = vars(ap.parse_args())

    if int(args["flag"]):
        obj = CreateCSV()

        # sample_size take integer value
        obj.generate_csv(sample_size=obj.sample_size)
    else:
        """
        output image with Bounding Box
        """
        try:
            img = args["image"]
            img = cv2.imread(img)
            img = transforms.ToPILImage()(img)
            img_pil = transforms.Resize((512, 512))(img)
            img = np.array(img_pil)
            xmin, ymin, xmax, ymax = int(args["bbox"][0]), int(args["bbox"][1]), int(args["bbox"][2]), int(args["bbox"][3])
            y = ymin - 10 if ymin - 10 > 10 else ymin + 10
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          (0, 100, 0), 2)
            cv2.putText(img, "mask", (xmin, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 2)

            cv2.imshow("Image", img)
            cv2.waitKey(0)
        except:
            print("Error!")
            print("     Image_path/Bounding_box Information Absent!")
            print("     Use the following command format:")
            print("     python3 create_dataset.py -f 0 -i '<face_image_path.{png|jpeg|jpg|*}>' --bbox xmin, ymin, xmax,"
                  "ymax")


if __name__ == '__main__':
    main()
