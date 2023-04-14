#!/usr/bin/python3

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import *


# def computeDisparityMap(img1, img2, F):
#     """
#     Compute the disparity map using the fundamental matrix.
#     ## Args:
#         stereo images: input images
#         F: Best fundamental calculated
#     ## Returns:
#         Normalized disparity map
#     """
#     # Convert to grayscale.
#     image_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     image_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     # Set the window size.
#     window = 7

#     # Initialize the left and right stero arrays
#     left_arr, right_arr = image_1, image_2
#     left_arr, right_arr = left_arr.astype(int), right_arr.astype(int)

#     # CHECK
#     if left_arr.shape != right_arr.shape:
#         raise Exception("Left- Right image shape do not match")
    
#     # Get the h and w of the image
#     h, w = left_arr.shape

#     # Initialize the disparity map
#     disparity_map = np.zeros((h,w))

#     #Compute disparity of each pixel.
#     x_new = w - (2*window)
#     for y in tqdm(range(window, h-window)):
#         block_left_arr = []
#         block_right_arr = []
#         for x in range(window, w-window):
#             block_left = left_arr[y:y + window,
#                                     x:x + window]
#             block_left_arr.append(block_left.flatten())

#             block_right = right_arr[y:y + window,
#                                     x:x + window]
#             block_right_arr.append(block_right.flatten())

#         block_left_arr = np.array(block_left_arr)
#         block_left_arr = np.repeat(block_left_arr[:, :, np.newaxis], x_new, axis=2)

#         block_right_arr = np.array(block_right_arr)
#         block_right_arr = np.repeat(block_right_arr[:, :, np.newaxis], x_new, axis=2)
#         block_right_arr = block_right_arr.T

#         abs_diff = np.abs(block_left_arr - block_right_arr)
#         sum_abs_diff = np.sum(abs_diff, axis = 1)
#         idx = np.argmin(sum_abs_diff, axis = 0)
#         disparity = np.abs(idx - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
#         disparity_map[y, 0:x_new] = disparity 
#         # Convert to uint8
#         disparity_map_int = np.uint8(disparity_map * 255 / np.max(disparity_map))

#     return disparity_map_int

def ncc_score(block_left, block_right):
    """Compute the normalized cross-correlation (NCC) score between two blocks."""
    mean_left = np.mean(block_left)
    mean_right = np.mean(block_right)
    std_left = np.std(block_left)
    std_right = np.std(block_right)
    ncc = np.sum((block_left - mean_left) * (block_right - mean_right)) / (std_left * std_right * block_left.size)
    return ncc

def non_overlapping_block_matching(left_img, right_img, block_size=5):
    """Compute the disparity map using non-overlapping block matching."""
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Pad the images to ensure that the blocks can be extracted from the edges
    pad_size = block_size // 2
    left_img_padded = np.pad(left_img, pad_size, mode='constant', constant_values=0)
    right_img_padded = np.pad(right_img, pad_size, mode='constant', constant_values=0)
    # Initialize the disparity map
    disparity_map = np.zeros_like(left_img)
    # Compute the disparity for each block in the left image
    for i in tqdm(range(pad_size, left_img.shape[0] + pad_size, block_size)):
        for j in range(pad_size, left_img.shape[1] + pad_size, block_size):
            # Extract the current block in the left image
            block_left = left_img_padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            # Search for the best matching block in the right image
            max_ncc = -1
            for k in range(pad_size, right_img.shape[1] + pad_size - block_size + 1, block_size):
                # Extract the current block in the right image
                block_right = right_img_padded[i-pad_size:i+pad_size+1, k-pad_size:k+pad_size+1]
                # Compute the NCC score between the two blocks
                ncc = ncc_score(block_left, block_right)
                # Update the best matching block if the NCC score is higher than the current maximum
                if ncc > max_ncc:
                    max_ncc = ncc
                    disparity = k - j
            # Update the disparity map with the computed disparity
            disparity_map[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1] = disparity
    # Normalize the disparity map from 0 to 255
    disparity_map_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return disparity_map_normalized


def get_images(image_paths, scale_factor=1.0):
    """
    Read in the images
    ## Returns:
        images: a list of the images
    """
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if scale_factor != 1.0:
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        images.append(img)
    return images


def estimate_fundamental_matrix(points1, points2):
    """
    Calculate the fundamental matrix from the given correspondences (at least 8).
    """ 
    # form the A matrix
    num_points = min(points1.shape[0], points2.shape[0])
    A = np.empty((num_points, 9))
    
    for i in range(num_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A[i] = np.array([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    # find SVD 
    U, S, Vh = np.linalg.svd(A)
    
    # F is given by the column of Vh that has the smallest singular value
    F = np.reshape(Vh[-1], (3, 3))
   
    # force F to be singular by setting smallest singular value to zero
    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vh



    return F


def fundamental_ransac(correspondences, iterations = 2000, threshold = 0.01):
    """
    Compute the fundamental matrix using RANSAC.
    """
    points1 = correspondences[0]
    points2 = correspondences[1]
    points1_homogenous = np.append(points1, np.ones((points1.shape[0], 1)), axis=1)
    points2_homogenous = np.append(points2, np.ones((points2.shape[0], 1)), axis=1)
    
    best_inliers = None
    best_F = None
    most_inliers = 0
    
    for _ in range(iterations):
        # choose 8 random points
        rand_indexes = np.random.choice(points1.shape[0], 8, replace=False)
        rand_points1 = points1[rand_indexes]
        rand_points2 = points2[rand_indexes]
              
        # find F
        F = estimate_fundamental_matrix(rand_points1, rand_points2)
        
        # count inliers (use fact that a*F*b.T = 0)
        residuals = np.empty((points1_homogenous.shape[0], ))
        for i in range(residuals.shape[0]):
            residuals[i] = np.abs(points1_homogenous[i] @ F @ points2_homogenous[i].T)

        num_inliers = np.sum(residuals < threshold)
        
        # record if best
        if num_inliers > most_inliers:
            most_inliers = num_inliers
            best_F = F
            # build correspondences
            best_inliers = (points1[residuals < threshold], points2[residuals < threshold])

    # re-estimate F using best inliers
    best_F = estimate_fundamental_matrix(*best_inliers)

    return best_F, best_inliers


def get_nonmax_suppression(img, window_size=3):
    """
    Apply non-maximum suppression to an image

    ## Returns:
        img_copy: a copy of the image with non-maximum suppression applied
    """
    img_copy = img.copy()
    img_min = img.min()

    for r, c in np.ndindex(img_copy.shape):
        # get window around specific pixel
        c_lower = max(0, c-window_size//2)
        c_upper = min(img_copy.shape[1], c+window_size//2)
        r_lower = max(0, r-window_size//2)
        r_upper = min(img_copy.shape[0], r+window_size//2)
        
        # set pixel to img_min so it is not included in max calculation
        temp = img_copy[r, c]
        img_copy[r, c] = img_min

        # if pixel is the max in the window, keep it, otherwise keep it img_min
        if temp > img_copy[r_lower:r_upper, c_lower:c_upper].max():
            img_copy[r, c] = temp
    
    return img_copy


def get_harris_corners(img, num_corners=1000, window_size=5, neighborhood_size=7):
    """
    Detect Harris corners in an image, returning their locations and neighborhoods

    ## Returns:
        corners: (num_corners, 2) array of (x, y) coordinates of the corners
        neighborhood: (num_corners, neighborhood_size, neighborhood_size) array of the neighborhoods around the corners
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # calulate derivatives
    Ix = cv2.Sobel(img_gray, ddepth=-1, dx=1, dy=0, ksize=3)
    Iy = cv2.Sobel(img_gray, ddepth=-1, dx=0, dy=1, ksize=3)

    # derivative products
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # sum of products 
    sum_kernel = np.ones((window_size, window_size))
    Sxx = cv2.filter2D(src=Ixx, ddepth=-1, kernel=sum_kernel)
    Syy = cv2.filter2D(src=Iyy, ddepth=-1, kernel=sum_kernel)
    Sxy = cv2.filter2D(src=Ixy, ddepth=-1, kernel=sum_kernel)

    # calculate C matricies and R values
    R = np.empty(shape=Sxx.shape, dtype=np.float32)
    for i, j in np.ndindex(Sxx.shape):
        # set edges to zero as we cannot give them features easily
        if i < neighborhood_size//2 or i >= (R.shape[0] - neighborhood_size//2) or j < neighborhood_size//2 or j >= (R.shape[1] - neighborhood_size//2):
            R[i, j] = 0
            continue
        # calculate R value
        C = np.array([[Sxx[i, j], Sxy[i, j]], [Sxy[i, j], Syy[i, j]]])
        R[i, j] = np.linalg.det(C) - 0.04 * (np.trace(C) ** 2)

    # Calculate Non-maximum suppression
    Rs = get_nonmax_suppression(R)

    # Display the images in a single window (debugging)
    Ixx_disp = cv2.normalize(Ixx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Iyy_disp = cv2.normalize(Iyy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    R_disp = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Rs_disp = cv2.normalize(Rs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Hor1 = cv2.hconcat([Ixx_disp, Iyy_disp])
    Hor2 = cv2.hconcat([R_disp, Rs_disp])
    cv2.imshow("Harris Corner Detection (Gradient and Response visualization)", cv2.vconcat([Hor1, Hor2]))
    cv2.imwrite("output_harris.jpg", cv2.vconcat([Hor1, Hor2]))

    # Exception if the number of corners is greater than the number of pixels in the image
    if num_corners > Rs.size:
        raise ValueError("num_corners must be less than the number of pixels in the image")
    # Return the top num_corners corners by sorting and returning the indices
    corners = np.unravel_index(np.argsort(Rs, axis=None)[-num_corners:], R.shape)
    corners = np.stack((corners[1], corners[0]), axis=1)

    # Get the neighborhoods of the corners
    neighborhoods = np.empty((num_corners, neighborhood_size, neighborhood_size, img.shape[2]))
    for i, (x, y) in enumerate(corners):
        neighborhoods[i] = img[y-neighborhood_size//2:y+neighborhood_size//2+1, x-neighborhood_size//2:x+neighborhood_size//2+1]

    return corners, neighborhoods


def get_correspondences(corners1, neighborhoods1, corners2, neighborhoods2, max_correspondences_per_feature=5):
    """
    Find correspondences between the two images, returned as a dictionary mapping the corners
    from image1 to the corners in image2
    
    ## Returns:
        corr_a, corr_b: Two arrays of points where the same indexes correspond
    """
    corr_a = []
    corr_b = []
    # normalize neighborhoods
    neighborhoods1 -= neighborhoods1.mean(axis=0, keepdims=True)
    neighborhoods1 /= np.linalg.norm(neighborhoods1, axis=0)

    neighborhoods2 -= neighborhoods2.mean(axis=0, keepdims=True)
    neighborhoods2 /= np.linalg.norm(neighborhoods2, axis=0)

    # magic
    target_features_hashes = []
    for c1, n1 in zip(corners1, neighborhoods1):
        best_corner = (corners2[0], 0)
        for c2, n2 in zip(corners2, neighborhoods2):
            corr = np.sum(n1 * n2)
            if corr > best_corner[1]:
                best_corner = (c2, corr)

        best_corner_hash = hash(best_corner[0].tobytes())
        if target_features_hashes.count(best_corner_hash) < max_correspondences_per_feature:
            target_features_hashes.append(best_corner_hash)
            corr_a.append(c1)
            corr_b.append(best_corner[0])

    return np.array(corr_a), np.array(corr_b)


##########################
# Display helper functions
##########################


def display_harris_corners(img1, corners1, img2=None, corners2=None):
    """
    Display the Harris corners on top of the image
    """
    img1_copy = img1.copy()
    for corner in corners1:
        cv2.circle(img1_copy, corner, 2, (0, 0, 255), -1)
    if img2 is not None:
        img2_copy = img2.copy()
        for corner in corners2:
            cv2.circle(img2_copy, corner, 2, (0, 0, 255), -1)
        cv2.imshow("harris corners", np.concatenate((img1_copy, img2_copy), axis=1))
        cv2.imwrite("output_harris_corners.jpg", np.concatenate((img1_copy, img2_copy), axis=1))
    else:
        cv2.imshow("harris corners", img1_copy)
        cv2.imwrite("output_harris_corners.jpg", img1_copy)


def display_correspondences(img1, img2, correspondences, inliers=None):
    """
    Display the correspondences between the two images one on top of the other with lines
    """
    if inliers is None:
        color_func = lambda: np.random.uniform(0, 255, (3,))
    else:
        color_func = lambda: (0, 0, 255)
        
    images = np.concatenate((img1, img2), axis=1) 
    for (c1r, c1c), (c2r, c2c) in zip(correspondences[0], correspondences[1]):
        cv2.circle(images, (c1r, c1c), 2, (255, 0, 0), -1)
        cv2.circle(images, (c2r+img1.shape[1], c2c), 2, (255, 0, 0), -1)
        cv2.line(images, (c1r, c1c), (c2r+img1.shape[1], c2c), thickness=1, color=color_func())
    if inliers is not None:
        for (c1r, c1c), (c2r, c2c) in zip(inliers[0], inliers[1]):
            cv2.line(images, (c1r, c1c), (c2r+img1.shape[1], c2c), thickness=1, color=(0, 255, 0))
        cv2.imshow("inliers", images)
        cv2.imwrite("output_inliers.jpg", images)
    else:
        cv2.imshow("correspondences", images)
        cv2.imwrite("output_correspondences.jpg", images)


def display_epipolar_lines(img1, img2, fundamental_matrix, correspondences, num_points=50):
    frame1 = img1.copy()
    frame2 = img2.copy()
    points1_homogenous = np.append(correspondences[0], np.ones((correspondences[0].shape[0], 1)), axis=1)[:num_points]
    points2_homogenous = np.append(correspondences[1], np.ones((correspondences[1].shape[0], 1)), axis=1)[:num_points]
        
    # left image
    lines = fundamental_matrix @ points1_homogenous.T
    for l in lines.T:
        # calculate points on line at x=0 and x=frame.shape[1]
        # line is defined by l[0]*x + l[1]*y + l[2] = 0
        p1 = (0, int(-l[2]/l[1]))
        p2 = (frame1.shape[1], int(-(l[0]*frame1.shape[1]+l[2])/l[1]))
        frame1 = cv2.line(frame1, p1, p2, (255, 0, 0))
    for p in points1_homogenous:
        cv2.circle(frame1, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

    # right image
    lines = fundamental_matrix.T @ points2_homogenous.T
    for l in lines.T:
        p1 = (0, int(-l[2]/l[1]))
        p2 = (frame2.shape[1], int(-(l[0]*frame2.shape[1]+l[2])/l[1]))
        frame2 = cv2.line(frame2, p1, p2, (255, 0, 0))
    for p in points2_homogenous:
        cv2.circle(frame2, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        
    cv2.imshow("Epipolar Lines", np.hstack([frame1, frame2]))
    cv2.imwrite("output_epipolar_lines.jpg", np.hstack([frame1, frame2]))

#######
# Main
#######


def main():
    # Read in the command line arguments
    parser = argparse.ArgumentParser(description='Process multiple image files')
    parser.add_argument('image_filenames', type=str, nargs='+', help='the filenames of the images to process')
    args = parser.parse_args()
    image_filenames = args.image_filenames

    # Read in the images
    img1, img2 = get_images(image_filenames, scale_factor=1.0)

    # ii. Apply Harris corner detector to both images: compute Harris R function over the
    # image, and then do non-maximum suppression to get a sparse set of corner features.
    corners1, neighborhoods1 = get_harris_corners(img1, num_corners=500, neighborhood_size=19)
    corners2, neighborhoods2 = get_harris_corners(img2, num_corners=500, neighborhood_size=19)
    display_harris_corners(img1, corners1, img2, corners2)

    # iii. For each corner feature in image 1, find the best matching corner feature in image 2
    correspondences = get_correspondences(corners1, neighborhoods1, corners2, neighborhoods2)
    display_correspondences(img1, img2, correspondences)

    # iv. Use RANSAC to find the fundamental matrix that best fits the correspondences
    fundmental_matrix, best_set_corresp = fundamental_ransac(correspondences)
    print("Fundamental Matrix: \n", fundmental_matrix)
    display_correspondences(img1, img2, correspondences, best_set_corresp)
    display_epipolar_lines(img1, img2, fundmental_matrix, correspondences)

    # v. Compute the disparity map
    # Convert to grayscale
    #disparity_map = non_overlapping_block_matching(img1, img2, 7)
    disparity_map = non_overlapping_block_matching(img1, img2, 3)

    plt.imshow(disparity_map, cmap='plasma_r')
    plt.show()


if __name__ == "__main__":
    main()

    cv2.waitKey(0)
    cv2.destroyAllWindows()