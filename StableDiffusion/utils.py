import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import seaborn as sns
sns.set_theme(style="ticks", palette="pastel", color_codes=True, font_scale=1.2)

# diffusion process equation
def diffusion_process(image, beta_t, mean=0, std=1):
    noise = np.random.normal(mean, std, image.shape)
    return np.sqrt(1 - beta_t) * image + np.sqrt(beta_t) * noise, noise

def inverse_diffusion_process(diffused_image, beta_t, noise):
    recovered_image = (diffused_image - np.sqrt(beta_t) * noise) / np.sqrt(1 - beta_t)
    return recovered_image

def convert_img_to_grayscale(image):
    image = np.array(image)
    
    # If the image is not grayscale, convert it to grayscale
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # Normalize the image to [0, 1] range
    image = image / 255.0

    return image

def show_image(image, figsize=(8, 8), title='Test Image', cmap=None):
    plt.figure(figsize=figsize)
    
    # Ensure the image is within the valid range
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image, 0, 1)  # Clipping for float images
    elif image.dtype == np.uint8:
        image = np.clip(image, 0, 255)  # Clipping for uint8 images
    
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.show()
    
def show_image_grid(images, titles, cmaps = None, figsize=(10, 5), spacing=0.1):
    num_images = len(images)
    
    if titles is None:
        titles = [f'Image {i+1}' for i in range(num_images)]
    
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        
        # Ensure the image is within the valid range
        if images[i].dtype == np.float32 or images[i].dtype == np.float64:
            images[i] = np.clip(images[i], 0, 1)  # Clipping for float images
        elif images[i].dtype == np.uint8:
            image = np.clip(image, 0, 255)  # Clipping for uint8 images
        
        cmap = cmaps[i] if cmaps and i < len(cmaps) else None
        ax.imshow(images[i], cmap=cmap)
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.subplots_adjust(wspace=spacing)
    plt.show()
    
def convert_image_to_array(image):
    image = np.array(image)
    image = image.flatten()
    return image

def open_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    return normalize_image(image)

def plot_histogram(image, num_bins=100, figsize=(10, 5)):
    image = np.array(image)
    image = image.flatten()
    plt.figure(figsize=figsize)
    plt.hist(image, bins=num_bins, range=(0, 1), edgecolor='black')
    plt.title('Histogram of Grayscale Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def create_gaussian_distribution(shape, mean=0, std=1):
    """
    Create a distribution of data with the given shape and transform it to a Gaussian distribution with mean 0 and std 1.
    """
    # Generate random data with the same shape
    data = np.random.normal(loc=mean, scale=std, size=shape)
    
    return data

def normalize_image(image):
    image = image / 255.0
    return np.clip(image, 0, 1)

