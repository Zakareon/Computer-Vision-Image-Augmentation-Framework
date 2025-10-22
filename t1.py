import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET

def load_images_from_directory():
    root = tk.Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="Select Input Directory") # Select directory with images
    
    files = os.listdir(input_dir) # Obtain image names from directory
    
    # Read each image from directory and add to list
    image_list = []
    for file in files:
        img = cv2.imread(os.path.join(input_dir + '/' + file))
        image_list.append(img)
    
    return (image_list, input_dir, files)

def load_augmentation_settings(config_file):
    tree = ET.parse(config_file)
    root = tree.getroot()

    augmentations_array = []
    for augmentation in root.findall('augmentation'):
        name = augmentation.get('name')
        params = [float(param.text) for param in augmentation.findall('param')]
        augmentation_data = (name, params)
        augmentations_array.append(augmentation_data)

    return augmentations_array


def display_images(images_list):
    cv2.imshow('Image 0', images_list[0])
    cv2.imshow('Image 1', images_list[1])
    cv2.imshow('Image 2', images_list[2])
    cv2.imshow('Image 3', images_list[3])
    cv2.imshow('Image 4', images_list[4])

def save_augmented_image(output_dir, image, original_name, aug_name, index):
    base_name = os.path.splitext(os.path.basename(original_name))[0]
    output_path = os.path.join(output_dir, f"{base_name}_{aug_name}_{index}.jpg")
    cv2.imwrite(output_path, image)

def apply_augmentation(image, augmentation):
    aug_name, param = augmentation
    if aug_name == "rotation":
        augmented_image = rotation(image, param[0])
    if aug_name == "brightness":
        augmented_image = brightness_adjustment(image, param[0])
    if aug_name == "contrast":
        augmented_image = contrast_adjustment(image, param[0])
        augmented_image = scale_image(augmented_image, 0.5)
    if aug_name == "gamma correction":
        augmented_image = gamma_correction(image, param[0])
    if aug_name == "histogram equalization":
        augmented_image = histogram_equalization(image)
    if aug_name == "flip":
        augmented_image = flip_image(image, param[0])
    if aug_name == "scale":
        augmented_image = scale_image(image, param[0])
    if aug_name == "color distortion":
        augmented_image = color_distortion(image, param[0], param[1], param[2])
    if aug_name == "gaussian noise addition":
        augmented_image = gaussian_noise_addition(image, param[0], param[1])
    if aug_name == "logarithmic point transformation":
        augmented_image = logarithmic_point_transformation(image)
    if aug_name == "box filter":
        augmented_image = box_filter(image, (int(param[0]), int(param[0])))
    if aug_name == "gaussian filter":
        augmented_image = gaussian_filter(image, (int(param[0]), int(param[0])), int(param[1]))
    if aug_name == "bilateral filter":
        augmented_image = bilateral_filter(image, int(param[0]), int(param[1]), int(param[2]))
    if aug_name == "median filter":
        augmented_image = median_filter(image, int(param[0]))
    if aug_name == "high-pass filter":
        augmented_image = high_pass_filter(image, int(param[0]), int(param[1]))
    if aug_name == "grayscale":
        augmented_image = grayscale_conversion(image)
    if aug_name == "binary conversion":
        augmented_image = binary_conversion(image, int(param[0]))
    if aug_name == "translate":
        augmented_image = translate_image(image, int(param[0]), int(param[1]))
    if aug_name == "shear":
        augmented_image = shear_image(image, int(param[0]), int(param[1]))
    return augmented_image

def brightness_adjustment(image, value):
    image_array = image.astype(np.float64)
    image_array += value
    image_array = np.clip(image_array, 0, 255)

    return image_array.astype(np.uint8)

def contrast_adjustment(image, contrast):
    contrast = 2.0
    img_array = image.astype(np.float64)
    img_array = 128 + contrast * (img_array - 128)
    img_array = np.clip(img_array, 0, 255)

    return img_array.astype(np.uint8)

def color_distortion(image, blue, green, red):
    distorted_image = np.array(image)
    distorted_image[:,:,0] = np.clip(distorted_image[:,:,0]*blue, 0, 255)
    distorted_image[:,:,1] = np.clip(distorted_image[:,:,1]*green, 0, 255)
    distorted_image[:,:,2] = np.clip(distorted_image[:,:,2]*red, 0, 255)
    
    return distorted_image.astype(np.uint8)

def gamma_correction(image, gamma):
    image_array = np.array(image)
    normalized_image = (image_array)/255.0
    corrected_image = np.power(normalized_image, gamma)

    return np.clip(corrected_image*255, 0, 255).astype(np.uint8)

def histogram_equalization(image):
    image_array = np.array(image)
    hist, bins = np.histogram(image_array.flatten(), 256, (0,256))
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    equalized_image = cdf_final[image_array]

    return equalized_image

def flip_image(image, direction):
    image_array = np.array(image)
    if direction == 1:
        flipped_image = np.flip(image_array, 1)
    elif direction == 2:
        flipped_image = image_array.transpose((1,0,2))

    return flipped_image

def scale_image(image, scale_factor):
    image_array = np.array(image)
    h, w = image_array.shape[:2]
    new_h = int(h*scale_factor)
    new_w = int(w*scale_factor)
    scaled_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # nearest neighbour interpolation
    for i in range(new_h):
        for j in range(new_w):
            x = int(j / scale_factor)
            y = int(i / scale_factor)

            x = min(x, w - 1)
            y = min(y, h - 1)

            scaled_image[i, j] = image_array[y, x]
    
    return scaled_image

def rotation(image, param):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, param, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    return rotated_image

def color_distortion(image, blue=1.0, green=1.0, red=1.0):
    distorted_image = np.array(image)
    distorted_image[:,:,0] = np.clip(distorted_image[:,:,0]*blue, 0, 255)
    distorted_image[:,:,1] = np.clip(distorted_image[:,:,1]*green, 0, 255)
    distorted_image[:,:,2] = np.clip(distorted_image[:,:,2]*red, 0, 255)
    
    return distorted_image.astype(np.uint8)

def gaussian_noise_addition(image, mean=0, std=0):
    image_array = np.array(image)
    gaussian_noise = np.random.normal(mean, std, image_array.shape).astype(np.uint8)
    noisy_image = np.clip(image_array + gaussian_noise, 0, 255)

    return noisy_image.astype(np.uint8)

def logarithmic_point_transformation(image):
    image_array = np.float32(image)
    c = 255 / np.log(1 + np.max(image_array))
    log_image = c * np.log(1 + image_array)
    log_image = np.clip(log_image, 0, 255)

    return log_image.astype(np.uint8)

def box_filter(image, kernel_size=(5, 5)):
    filtered_image = cv2.boxFilter(image, ddepth=-1, ksize=kernel_size)
    return filtered_image

def gaussian_filter(image, kernel_size=(5,5), sigma=0):
    filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return filtered_image

def bilateral_filter(image, diameter=15, sigma_color=75, sigma_space=75):
    filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return filtered_image

def median_filter(image, kernel=5):
    filtered_image = cv2.medianBlur(image, kernel)
    return filtered_image

def high_pass_filter(image, kernel_size=21, sigma=3):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low_pass = gaussian_filter(gray_image, (kernel_size, kernel_size), sigma)
    high_pass_image = cv2.subtract(gray_image, low_pass)
    return high_pass_image

def grayscale_conversion(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def binary_conversion(image, threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i,j] > threshold:
                gray_image[i,j] = 255
            else:
                gray_image[i,j] = 0
    return gray_image

def translate_image(image, tx, ty):
    original_array = np.array(image)
    height, width = original_array.shape[:2]
    translated_image = np.zeros_like(original_array)

    # Loop through each pixel in the original image
    for y in range(height):
        for x in range(width):
            new_x = x + tx
            new_y = y + ty
            
            # Check if new coordinates are within bounds
            if 0 <= new_x < width and 0 <= new_y < height:
                translated_image[new_y, new_x] = original_array[y, x]

    return translated_image

def shear_image(image, shear_factor_x=0.5, shear_factor_y=0.5):
    image_array = np.array(image)

    height, width = image_array.shape[:2]

    new_w = int(width + height * shear_factor_x)
    new_h = int(height + width * shear_factor_y)
    sheared_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            new_x = int(x + y * shear_factor_x)
            new_y = int(y + x * shear_factor_y)

            if 0 <= new_x <= new_w and 0 <= new_y <= new_h:
                sheared_image[new_y, new_x] = image_array[y, x]

    return sheared_image

def main():
    print("\n************************************  START  ************************************\n")

    # Read images from directory
    (images_list, input_dir, files) = load_images_from_directory()

    # Read augmentations from configuration file
    config_file = 'C:\\Users\\germa\\OneDrive\\Desktop\\Final project\\Week2\\augmentation_config.xml'
    
    # Load augmentations
    augmentations = load_augmentation_settings(config_file)

    # Create output directory
    output_dir = input_dir + "_aug"
    os.makedirs(output_dir, exist_ok=True)

    # Apply augmentation algorithms and save in output directory
    index = 1
    augmented_images_list = []
    for image, file in zip(images_list, files):
        for augmentation in augmentations:
            aug_name, param = augmentation
            augmented_image = apply_augmentation(image, augmentation)
            augmented_images_list.append(augmented_image)
            save_augmented_image(output_dir, augmented_image, file, aug_name, index)
            index += 1

    # print(augmentations)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n************************************  END  ************************************\n")

if __name__ == "__main__":
    main()