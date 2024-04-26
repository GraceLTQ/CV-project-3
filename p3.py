import numpy as np
from PIL import Image

from scipy import ndimage, signal


############### ---------- Basic Image Processing ------ ##############

# TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):
    # Open the image file using PIL
    img = Image.open(filename)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Convert the numpy array to float
    img_float = img_array.astype(np.float32)

    # Normalize the image data to 0-1 range
    img_normalized = img_float / 255.0

    if len(img_normalized[0, 0]) != 3:
        img_normalized = img_normalized[:, :, 0:3]

    return img_normalized


# TODO 2: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    #     ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    #     xx, yy = np.meshgrid(ax, ax)
    #     kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    #     kernel /= np.sum(kernel)
    #     return kernel

    center = k // 2
    filter = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            x = abs(center - j)
            y = abs(center - i)
            filter[i][j] = np.exp(-(x**2 + y**2) / (2 * (sigma**2)))

    filter /= np.sum(filter)
    return filter


# TODO 3: Compute the image gradient.
# First convert the image to grayscale by using the formula:
# Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
# Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise. (use scipy.signal.convolve)
# Convolve with [0.5, 0, -0.5] to get the X derivative on each channel and convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel. (use scipy.signal.convolve)
# Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):
    #     # Convert to grayscale if it's a color image
    #     grayscale = 0.2125 * img[:, :, 0] + 0.7154 * \
    #         img[:, :, 1] + 0.0721 * img[:, :, 2]

    #     # Apply Gaussian smoothing
    #     gaussian = gaussian_filter(5, 1)
    #     smoothed = signal.convolve(
    #         grayscale, gaussian, mode='same')

    #     # Define derivative kernels as 2D arrays
    #     kernel_x = np.array([[0.5, 0, -0.5]])
    #     kernel_y = np.array([[0.5], [0], [-0.5]])

    #     # Convolve to find the derivatives, using 'same' to keep the original image size
    #     dx = signal.convolve(smoothed, kernel_x, mode='same')
    #     dy = signal.convolve(smoothed, kernel_y, mode='same')

    #     # Calculate the magnitude and orientation of gradients
    #     magnitude = np.sqrt(dx ** 2 + dy ** 2)
    #     orientation = np.arctan2(dy, dx)

    #     return magnitude, orientation

    # Convert to grayscale if it's a color image
    Y = 0.2125 * img[:, :, 0] + 0.7154 * img[:, :, 1] + 0.0721 * img[:, :, 2]

    # Apply Gaussian smoothing
    smoothed = signal.convolve(Y, gaussian_filter(5, 1), mode="same")

    # Convolve to find the derivatives, using 'same' to keep the original image size

    dx = signal.convolve(smoothed, np.array([[0.5, 0, -0.5]]), mode="same")
    dy = signal.convolve(smoothed, np.array([[0.5], [0], [-0.5]]), mode="same")

    # Calculate the magnitude and orientation of gradients
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx)

    return magnitude, orientation


# ----------------Line detection----------------

# TODO 4: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
# x cos(theta) + y sin(theta) + c = 0
# The input x and y are arrays representing the x and y coordinates of each pixel
# Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    d = np.abs(np.cos(theta)*x + np.sin(theta)*y + c)
    return d < thresh


# TODO 5: Write a function to draw a set of lines on the image. The `lines` input is a list of (theta, c) pairs. Each line must appear as red on the final image
# where every pixel which is less than thresh units away from the line should be colored red
def draw_lines(img, lines, thresh):
    i = img.copy()
    h, w, _ = i.shape
    indices = np.arange(h*w)
    y, x = np.unravel_index(indices, (h, w))
    for (t, c) in lines:
        d = check_distance_from_line(x, y, t, c, thresh)
        d = np.reshape(d, (h, w))
        r, c = np.where(d)
        i[r, c, :] = [1, 0, 0]
    return i

#     # Ensure the image is in the right format, expecting a color image
#     if img.ndim != 3 or img.shape[2] != 3:
#         raise ValueError("Input image must be a color image with three channels (RGB).")

#     # Make a copy of the image to modify it
#     output_img = np.copy(img)

#     # Get image dimensions
#     height, width, _ = img.shape

#     # Iterate over all pixels in the image
#     for y in range(height):
#         for x in range(width):
#             for theta, c in lines:
#                 # Calculate the distance from the current pixel to the line described by theta and c
#                 distance = abs(x * np.cos(theta) + y * np.sin(theta) + c) / np.sqrt(np.cos(theta)**2 + np.sin(theta)**2)

#                 # If the distance is less than the threshold, set the pixel color to red
#                 if distance < thresh:
#                     output_img[y, x] = [1, 0, 0]  # Red in [0, 1] normalized RGB

#     return output_img


# TODO 6: Do Hough voting. You get as input the gradient magnitude and the gradient orientation, as well as a set of possible theta values and a set of possible c
# values. If there are T entries in thetas and C entries in cs, the output should be a T x C array. Each pixel in the image should vote for (theta, c) if:
# (a) Its gradient magnitude is greater than thresh1
# (b) Its distance from the (theta, c) line is less than thresh2, and
# (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    num_t = len(thetas)
    num_c = len(cs)

    # find indices of entries in gradmag that are > thresh1
    y_indices, x_indices = np.where(gradmag > thresh1)

    # output[a][b] = number votes for line with theta = thetas[a] and c = cs[b]
    output = np.zeros((num_t, num_c))

    for t_i, t in enumerate(thetas):
        for c_i, c in enumerate(cs):
            near_line = check_distance_from_line(
                x_indices, y_indices, t, c, thresh2)
            valid_orientation = np.abs(
                gradori[y_indices, x_indices]-t) < thresh3

            votes = np.sum(near_line & valid_orientation)
            output[t_i, c_i] = votes

    return output


# TODO 7: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if (a) its votes are greater than thresh, and
# (b) its value is the maximum in a (nbhd x nbhd) neighborhood in the votes array.
# Return a list of (theta, c) pairs
def localmax(votes, thetas, cs, thresh, nbhd):
    #     output = []
    #     nbhd_radius = nbhd // 2
    #     y_indices, x_indices = np.where(votes > thresh)

    #     for i in range(len(y_indices)):
    #         y = y_indices[i]
    #         x = x_indices[i]

    #         l = max(x - nbhd_radius, 0)
    #         r = min(x + nbhd_radius + 1, len(votes[0]))
    #         t = max(y - nbhd_radius, 0)
    #         b = min(y + nbhd_radius + 1, len(votes))

    #         neighborhood = votes[t:b, l:r]

    #         if votes[y, x] == np.max(neighborhood):
    #             output.append((thetas[y], cs[x]))

    #     return output

    # Initialize a list to store the coordinates of the local maxima
    local_maxima = []

    # Adjust neighborhood size for edge cases
    nbhd_radius = nbhd // 2

    # Iterate through each entry in the votes matrix
    for i in range(len(thetas)):
        for j in range(len(cs)):
            # Only consider votes greater than the threshold
            if votes[i, j] > thresh:
                # Define the neighborhood range
                min_i = max(i - nbhd_radius, 0)
                max_i = min(i + nbhd_radius + 1, votes.shape[0])
                min_j = max(j - nbhd_radius, 0)
                max_j = min(j + nbhd_radius + 1, votes.shape[1])

                # Extract the neighborhood
                neighborhood = votes[min_i:max_i, min_j:max_j]

                # Check if the current vote is the maximum in its neighborhood
                if votes[i, j] == np.max(neighborhood):
                    # If it's the highest and it's uniquely the highest in the neighborhood
                    #                     if np.count_nonzero(neighborhood == votes[i, j]) == 1:
                    local_maxima.append((thetas[i], cs[j]))

    return local_maxima


# Final product: Identify lines using the Hough transform
def do_hough_lines(filename):

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
    imgdiagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori, thetas, cs, 0.2, 0.5, np.pi/40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines
