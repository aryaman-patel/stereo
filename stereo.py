import cv2
import numpy as np
import argparse


def calculate_fundamental_matrix(points1, points2):
    """
    Calculate the fundamental matrix from the given 8 correspondences.
    """ 

    return F


def fundamental_ransac(correspondences, max_iter = 2000):
    """
    Compute the fundamental matrix using RANSAC.
    """

    return best_F

def get_nonmax_suppression(img, window_size=5):
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
        final_correspondences: dictionary mapping (x1, y1) to (x2, y2)
    """
    final_correspondences = {}
    
    # normalize neighborhoods
    print(neighborhoods1.shape)
    neighborhoods1 -= neighborhoods1.mean(axis=0, keepdims=True)
    neighborhoods1 /= np.linalg.norm(neighborhoods1, axis=0)

    neighborhoods2 -= neighborhoods2.mean(axis=0, keepdims=True)
    neighborhoods2 /= np.linalg.norm(neighborhoods2, axis=0)

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
            final_correspondences[tuple(c1)] = best_corner[0]

    return final_correspondences


def get_args():
    """
    Parse the command line arguments and return the two images
    """




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

    images = np.concatenate((img1, img2), axis=1) 
    for (c1r, c1c), (c2r, c2c) in correspondences.items():
        cv2.circle(images, (c1r, c1c), 2, (255, 0, 0), -1)
        cv2.circle(images, (c2r+img1.shape[1], c2c), 2, (255, 0, 0), -1)
        cv2.line(images, (c1r, c1c), (c2r+img1.shape[1], c2c), thickness=1, color=(0, 0, 255))
    if inliers is not None:
        for (c1r, c1c), (c2r, c2c) in inliers.items():
            cv2.line(images, (c1r, c1c), (c2r+img1.shape[1], c2c), thickness=1, color=(0, 255, 0))
    cv2.imshow("correspondences", images)
    cv2.imwrite("output_correspondences.jpg", images)


def main():
    # Read in the command line arguments
    parser = argparse.ArgumentParser(description='Process multiple image files')
    parser.add_argument('image_filenames', type=str, nargs='+', help='the filenames of the images to process')

    args = parser.parse_args()

    image_filenames = args.image_filenames

    # Read in the images
    img1 = cv2.imread(image_filenames[0])
    img2 = cv2.imread(image_filenames[1])

    # ii. Apply Harris corner detector to both images: compute Harris R function over the
    # image, and then do non-maximum suppression to get a sparse set of corner features.
    corners1, neighborhoods1 = get_harris_corners(img1, num_corners=500, neighborhood_size=19)
    corners2, neighborhoods2 = get_harris_corners(img2, num_corners=500, neighborhood_size=19)
    display_harris_corners(img1, corners1, img2, corners2)

    # iii. For each corner feature in image 1, find the best matching corner feature in image 2
    correspondences = get_correspondences(corners1, neighborhoods1, corners2, neighborhoods2)

    # iv. Use RANSAC to find the fundamental matrix that best fits the correspondences
    fundmental_matrix, best_set_corresp = fundamental_ransac(correspondences)
    print("Fundamental : \n", fundmental_matrix)
    display_correspondences(img1, img2, correspondences, best_set_corresp)

    # v. disparity map
    

if __name__ == "__main__":
    main()

    cv2.waitKey(0)
    cv2.destroyAllWindows()