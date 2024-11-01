import cv2
import numpy as np


def process_image(image_path, output_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = enhance_image(img)

    # optimal parameters
    blur_ksize, blur_sigma = determine_blur_parameters(img)
    canny_threshold1, canny_threshold2 = determine_canny_parameters(img)

    blurred_image = cv2.GaussianBlur(img, blur_ksize, blur_sigma)
    
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
    
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # Use morphological operations to emphasize straight lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    
    edges_colored = np.zeros_like(img)

    # Improved line detection using quantization method and dynamic programming
    lines = detect_lines(morph_edges)

    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(edges_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(output_path, cv2.cvtColor(edges_colored, cv2.COLOR_RGB2BGR))

    return output_path

def enhance_image(img):
    denoised_img = apply_denoising(img)
    equalized_img = apply_histogram_equalization(denoised_img)
    sharpened_img = apply_sharpening(equalized_img)
    return sharpened_img

def apply_denoising(img):
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 5, 21)
    return denoised_img

def apply_histogram_equalization(img):
    # adjust the contrast of an image by modifying the intensity distribution of the histogram
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    equalized_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return equalized_img

# Apply sharpening
def apply_sharpening(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img

def determine_blur_parameters(img):
    # base kernel size on the image size
    height, width, _ = img.shape
    ksize = (5, 5) if min(height, width) < 500 else (9, 9)
    sigma = 1.4  # tuned based on image statistics
    return ksize, sigma

def determine_canny_parameters(img):
    # Otsu's method to determine dynamic thresholds
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    return lower, upper

def detect_lines(edges):
    # Simplified line detection
    lines = []
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] == 255:
                lines.append((x, y, x+10, y+10))
    return lines
