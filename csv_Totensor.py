# from skimage import io, transform
import pandas as pd
from torchvision import transforms
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torchvision.transforms.functional as FT
import random
import utils
import math
import sys


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.
    Helps to learn to detect smaller objects.
    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """

    if (len(boxes.shape)) == 1:
        boxes = torch.tensor(np.expand_dims(np.array(boxes), axis=0))

    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def flip(image, boxes):
    """
    Flip image horizontally.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    if (len(boxes.shape)) == 1:
        boxes = torch.tensor(np.expand_dims(np.array(boxes), axis=0))

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=None):
    """
    Resize image. For the SSD300, resize to (300, 300).
    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)

    new_boxes = boxes / old_dims  # percent coordinates
    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def random_crop(image, boxes, labels):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.
    Note that some objects may be cut out entirely.
    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    if (len(boxes.shape)) == 1:
        boxes = torch.tensor(np.expand_dims(np.array(boxes), axis=0))

    if (len(labels.shape)) == 1:
        labels = torch.tensor(np.expand_dims(np.array(labels), axis=0))

    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels


def display_image(image, boxes):
    """
    Display the image with bounding box information
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    """
    image = np.array(image)
    xmin, ymin, xmax, ymax = int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3])
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                  (0, 100, 0), 2)
    cv2.imshow("BBox Image", image)
    cv2.waitKey(0)


def transform(image, boxes, labels, split):
    """
    Apply the transformations above.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'TRAIN', 'TEST'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels

    # Skip the following operations for evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        if (len(new_boxes.shape)) == 1:
            new_boxes = torch.tensor(np.expand_dims(np.array(boxes), axis=0))

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (512, 512) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(512, 512), return_percent_coords=True)

    # # display image with bounding box
    # display_image(new_image, new_boxes)

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, bbox = sample['image'], sample['boxes']
        #
        # image = image.transpose((2, 0, 1))
        # return {'image': torch.from_numpy(image),
        #         'boxes': torch.from_numpy(bbox)}

        image = sample

        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


class CSV2Tensor:

    def __init__(self, csv_file="datasets.csv", transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.split = "TRAIN"
        self.transform = transform

        self.image_path = []
        self.bndboxs = []
        self.labels = []

        for index, row in self.data_frame.iterrows():
            face_info = list(row)
            self.bndboxs.append(face_info[1: 5])
            self.labels.append(face_info[-1])
            self.image_path.append(face_info[0])

    def __getitem__(self, idx):
        bndboxs = torch.FloatTensor(self.bndboxs[idx])

        labels = np.array([self.labels[idx]])
        labels = torch.LongTensor(labels)

        img = cv2.imread(self.image_path[idx])
        img = transforms.ToPILImage()(img)
        img_pil = transforms.Resize((512, 512))(img)

        image, boxes, labels = transform(img_pil, bndboxs, labels, split=self.split)
        boxes, labels = boxes.squeeze(), labels.squeeze()

        target = {"boxes": boxes, "labels": labels}

        return image, target

    def __len__(self):
        return len(self.data_frame)


dataset = CSV2Tensor(transform=transforms.Compose([ToTensor()]))
# image, target = dataset[0]
# print(target)
# torch.manual_seed(1)


class DisplayBatchImages:

    def __init__(self, dataset=None):
        self.dataset = dataset

    def display(self):
        indices = torch.randperm(len(self.dataset)).tolist()

        train_no = int(len(indices) * 0.8)
        test_no = int(len(indices) - train_no)

        train_idx = indices[:train_no]
        test_idx = indices[-test_no:]

        dataset_train = torch.utils.data.Subset(self.dataset, train_idx)
        dataset_test = torch.utils.data.Subset(self.dataset, test_idx)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4,
                                                       shuffle=True, num_workers=0)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4,
                                                      shuffle=False, num_workers=0)

        """
        display the batch images for train dataset
        """
        for i_batch, sample_batched in enumerate(dataloader_train):

            image_batch, bbox_batch = sample_batched['image'], sample_batched['bbox']
            batch_size = len(image_batch)

            I = []
            for i in range(batch_size):
                image = image_batch[i].numpy()
                bbox = bbox_batch[i]
                label = sample_batched["labels"][i]
                print(label)
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                if int(label):
                    text = "no_mask"
                else:
                    text = "mask"
                y = ymin - 10 if ymin - 10 > 10 else ymin + 10
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                              (0, 100, 0), 2)
                cv2.putText(image, text, (xmin, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 2)
                img = Image.fromarray(image.astype('uint8'), 'RGB')
                img = img.resize((256, 256), Image.ANTIALIAS)
                img = np.array(img)
                I.append(img)

            img_concate_Hori = np.concatenate(tuple(I), axis=1)
            cv2.imshow('Train Bounding Box Images', img_concate_Hori)
            break

        """
        display the batch images for test dataset
        """
        for i_batch, sample_batched in enumerate(dataloader_test):

            image_batch, bbox_batch = sample_batched['image'], sample_batched['bbox']
            batch_size = len(image_batch)

            I = []
            for i in range(batch_size):
                image = image_batch[i].numpy()
                bbox = bbox_batch[i]
                label = sample_batched["labels"][i]
                print(label)
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                if int(label):
                    text = "no_mask"
                else:
                    text = "mask"
                y = ymin - 10 if ymin - 10 > 10 else ymin + 10
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                              (0, 100, 0), 2)
                cv2.putText(image, text, (xmin, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 2)
                img = Image.fromarray(image.astype('uint8'), 'RGB')
                img = img.resize((256, 256), Image.ANTIALIAS)
                img = np.array(img)
                I.append(img)

            img_concate_Hori = np.concatenate(tuple(I), axis=1)
            cv2.imshow('Test Bounding Box Images', img_concate_Hori)
            break

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# dis = DisplayBatchImages(dataset)
# dis.display()

"""
training steps
"""

indices = torch.randperm(len(dataset)).tolist()

train_no = int(len(indices) * 0.8)
test_no = int(len(indices) - train_no)

train_idx = indices[:train_no]
test_idx = indices[-test_no:]

dataset_train = torch.utils.data.Subset(dataset, train_idx)
dataset_test = torch.utils.data.Subset(dataset, test_idx)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4,
                                               shuffle=True, num_workers=0)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4,
                                              shuffle=False, num_workers=0)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


class Averages:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


loss_hist = Averages()
itr = 1
num_epochs = 20
print_freq = 10

"""
Training
"""
for epoch in range(num_epochs):
    loss_hist.reset()

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(dataloader_train) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # for idx, sample_batchs in enumerate(dataloader_train):
    for images, targets in metric_logger.log_every(dataloader_train, print_freq, header):

        # images = sample_batchs[0]
        # targets = sample_batchs[1]

        # images, targets = next(iter(dataloader_train))
        images = images.to(device)

        box_labels = []
        for idx in range(len(targets["boxes"])):
            box = np.expand_dims(np.array(targets["boxes"][idx]), axis=0)
            label = np.expand_dims(np.array(targets["labels"][idx]), axis=0)
            box_labels.append({"boxes": torch.tensor(box), "labels": torch.tensor(label)})

        targets = [{k: v.to(device) for k, v in t.items()} for t in box_labels]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()  # targets = [{k: v for k, v in t.items()} for t in targets]

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        loss_hist.send(loss_value)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print(f"Epoch #{epoch} loss: {loss_hist.value}")

    if epoch > 10:
        x = str(input("Enter 'Yes' to continue training for epoch[{}]: ".format(epoch)))
        if x.lower() == "yes":
            # saving the train model
            torch.save(model.state_dict(), "save_model.pt")
            print("training continues.....")
        else:
            break

print("training ends.....")

# saving the train model
torch.save(model.state_dict(), "save_model.pt")

"""
Testing
"""
