import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import argparse

matplotlib.use('TkAgg') # for interactive plotting

def inlier_outlier_benchmark(keypoints1, keypoints2, good_matches, K, threshold=5.0):
    """
    Estimate essential matrix from matched keypoints using RANSAC
    and calculates the inlier ratio based on the inlier mask
    """

    # at least 5 points to estimate essential matrix
    if len(good_matches) < 5:
        return None, None, 0.0

    # Extract keypoints for essential matrix calculation
    points1 = []
    points2 = []
    for match in good_matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        points1.append(pt1)
        points2.append(pt2)
    points1 = np.float32(points1).reshape(-1, 1, 2)
    points2 = np.float32(points2).reshape(-1, 1, 2)

    # Compute essential matrix with RANSAC to filter outliers
    E, mask = cv.findEssentialMat(points1, points2, K, cv.RANSAC, threshold=threshold)
    inliers_mask = mask.ravel().tolist()

    # Accurcy results
    num_matches = len(good_matches)
    num_inliers = np.sum(inliers_mask)
    inlier_ratio = (num_inliers / num_matches) * 100
    
    print("\n--- Benchmarking Results (RANSAC) ---")
    print(f"Total initial matches (after ratio test): {num_matches}")
    print(f"Number of RANSAC Inliers: {num_inliers}")
    print(f"Inlier Ratio: {inlier_ratio:.2f}%")
    
    return E, inliers_mask, inlier_ratio

def main(img1_path, img2_path):
    img1_grayscale = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    img2_grayscale = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
    img1_bgr = cv.imread(img1_path)
    img2_bgr = cv.imread(img2_path)

    # Camera intrinsics (res=640x480, fov=90)
    focal_length = 268.5 # f = W / (2*tan(fov/2))
    principal_point = (320, 240) 
    K = np.array([
        [focal_length, 0, principal_point[0]],
        [0, focal_length, principal_point[1]],
        [0, 0, 1]
    ])

    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=2000)

    # Find keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img1_grayscale, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_grayscale, None)

    # Match descriptors
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter poor matches
    good_matches = []
    ratio_thresh = 0.75
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Run RANSAC + Benchmarking
    E, inliers_mask, inlier_ratio = inlier_outlier_benchmark(
        keypoints1, keypoints2, good_matches, K
    )
    
   # Visualize keypoints and inlier matches
    if E is not None:
        img_inliers = cv.drawMatches(img1_bgr, keypoints1, img2_bgr, keypoints2, 
                                    good_matches, None, matchesMask=inliers_mask,
                                    flags=2) # only draw inliers and not draw single kp
    else:
        print("RANSAC failed, drawing all 'good_matches' instead.")
        img_inliers = cv.drawMatches(img1_bgr, keypoints1, img2_bgr, keypoints2,
                                    good_matches, None, flags=2)

    plt.figure(figsize=(10,6))
    plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
    plt.title("Matches keypoints")
    plt.axis('off')  
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=str)
    parser.add_argument('img2', type=str)
    args = parser.parse_args()

    main(args.img1, args.img2)