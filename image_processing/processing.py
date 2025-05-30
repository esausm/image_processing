import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

class Image:
    @staticmethod
    def find_difference(image1, image2):    
        if(image1.shape != image2.shape):
            raise Exception('Specify 2 images with de same shape.')

        gray_image1 = rgb2gray(image1)
        gray_image2 = rgb2gray(image2)
        
        score, diff_image = ssim(gray_image1, gray_image2, full=True)
        print(f'Similarity of the images: {score:.4f}')

        diff_image_normalized = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())
        return diff_image_normalized
    

    @staticmethod
    def transfer_histogram(image1, image2):
        matched_image = match_histograms(image1, image2, multichannel=True)
        return matched_image

    @staticmethod
    def resize(image, proportion):
        if(proportion < 0 or proportion > 1):
            raise Exception('Specify a valid proportion between 0 and 1.')
        height = round(image.shape[0] * proportion)
        width = round(image.shape[1] * proportion)
        return resize(image, (height, width), anti_aliasing=True)
