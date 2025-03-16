import os
import cv2
import random
from imgaug import augmenters as iaa
from data_scraper import MONGOLIAN_ALPHABET


def mix_images(img1, img2):
    """
    Blend two images together using a random alpha value.
    The second image is resized to match the first.
    """
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    alpha = random.uniform(0.3, 0.7)
    mixed = cv2.addWeighted(img1, alpha, img2_resized, 1 - alpha, 0)
    return mixed


def main(alphabet: str):
    folder = f'./dataset/{alphabet}/'
    # List all images in the folder
    images_list = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images_list:
        print(f"No images found in folder: {folder}")
        return

    # Geometric transformations: flip, crop, rotate, stretch, and zoom
    geo_aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Crop(percent=(0, 0.1)),
        iaa.Affine(
            rotate=(-45, 45),
            shear=(-16, 16),
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}
        )
    ])

    # Color space transformations: change hue/saturation, brightness, and contrast
    color_aug = iaa.Sequential([
        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
        iaa.Multiply((0.8, 1.2)),               # adjust brightness
        iaa.ContrastNormalization((0.75, 1.5))  # change contrast
    ])

    # Kernel filters: change sharpness and apply blurring
    kernel_aug = iaa.Sequential([
        iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 3.0))
    ])

    # Random erasing: randomly drop parts of the image
    erase_aug = iaa.CoarseDropout(p=0.05, size_percent=0.1, per_channel=0.5)

    # Number of augmentations per type
    num_geo = 3
    num_color = 3
    num_kernel = 3
    num_erase = 3
    num_mix = 3

    for image_name in images_list:
        image_path = os.path.join(folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read {image_name}. Skipping...")
            continue

        base_name = os.path.splitext(image_name)[0]

        # Geometric augmentation
        for i in range(num_geo):
            try:
                image_aug = geo_aug(image=image)
                new_name = f"{base_name}_geo_{i}.jpg"
                cv2.imwrite(os.path.join(folder, new_name), image_aug)
            except Exception as e:
                print(f"Error during geometric augmentation on {image_name}: {e}")

        # Color space augmentation
        for i in range(num_color):
            try:
                image_aug = color_aug(image=image)
                new_name = f"{base_name}_color_{i}.jpg"
                cv2.imwrite(os.path.join(folder, new_name), image_aug)
            except Exception as e:
                print(f"Error during color augmentation on {image_name}: {e}")

        # Kernel filters augmentation
        for i in range(num_kernel):
            try:
                image_aug = kernel_aug(image=image)
                new_name = f"{base_name}_kernel_{i}.jpg"
                cv2.imwrite(os.path.join(folder, new_name), image_aug)
            except Exception as e:
                print(f"Error during kernel augmentation on {image_name}: {e}")

        # Random erasing augmentation
        for i in range(num_erase):
            try:
                image_aug = erase_aug(image=image)
                new_name = f"{base_name}_erase_{i}.jpg"
                cv2.imwrite(os.path.join(folder, new_name), image_aug)
            except Exception as e:
                print(f"Error during erasing augmentation on {image_name}: {e}")

        for i in range(num_mix):
            try:
                other_image_name = random.choice(images_list)
                # Ensure the other image is different when possible
                while other_image_name == image_name and len(images_list) > 1:
                    other_image_name = random.choice(images_list)
                other_image_path = os.path.join(folder, other_image_name)
                other_image = cv2.imread(other_image_path)
                if other_image is None:
                    continue
                image_aug = mix_images(image, other_image)
                new_name = f"{base_name}_mix_{i}.jpg"
                cv2.imwrite(os.path.join(folder, new_name), image_aug)
            except Exception as e:
                print(f"Error during mixing augmentation on {image_name}: {e}")


if __name__ == "__main__":
    for alphabet in MONGOLIAN_ALPHABET:
        print(f"Augmenting images for alphabet: {alphabet.upper()}")
        main(alphabet.upper())
