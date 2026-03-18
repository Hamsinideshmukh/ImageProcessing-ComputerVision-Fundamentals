import cv2
import numpy as np
import sys

IMG1_PATH = r"C:\Users\Dell\OneDrive\Desktop\porfolio\q11.jpg"       # LEFT image
IMG2_PATH = r"C:\Users\Dell\OneDrive\Desktop\porfolio\q22.jpg"      # RIGHT image
OUTPUT_PATH = "panorama.jpg" 

RATIO = 0.85                 # Lowe's ratio test threshold (lower = stricter matching)
MIN_MATCH = 10               # minimum good keypoint matches needed
SMOOTHING_WINDOW = 800       # blend zone width in pixels (bigger = softer seam)



class Image_Stitching():
    def __init__(self):
        self.ratio = RATIO
        self.min_match = MIN_MATCH
        self.smoothing_window_size = SMOOTHING_WINDOW

        self.sift = cv2.SIFT_create()

    def registration(self, img1, img2):
        """
        Finds homography matrix H that maps img2 onto img1's coordinate space.
        Steps: detect keypoints , match them , filter good ones(thru RANSAC - fits a mathematical model only with relavent params ignoring the outliers) , compute H
        """
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        matcher = cv2.BFMatcher()  #Bruteforce matcher
        raw_matches = matcher.knnMatch(des1, des2, k=2)

        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            # Lowe's ratio test, we keep match only if it's clearly better than 2nd best
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])

        # Save a debug image showing matched keypoints between the two images
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)

        if len(good_points) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            # RANSAC robustly computes the 3x3 homography matrix
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        else:
            print(f"Not enough matches: {len(good_points)} found, need {self.min_match}")
            sys.exit(1)

        return H

    def create_mask(self, img1, img2, version):
        
        #Creates a float mask (values 0.0 to 1.0) for smooth blending at the seam.
       
        height_img1 = img1.shape[0]
        width_img1  = img1.shape[1]
        width_img2  = img2.shape[1]

        height_panorama = height_img1
        width_panorama  = width_img1 + width_img2

        offset  = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)

        mask = np.zeros((height_panorama, width_panorama),dtype=np.float32)

        if version == 'left_image':
            mask[:, barrier - offset : barrier + offset] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, barrier + offset:] = 1

        # Expand to 3 channels so it can multiply an RGB image
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        """
        Full pipeline: register → mask → warp → blend → crop.
        Returns the final stitched panorama as a numpy array.
        """
        self.smoothing_window_size = int(img1.shape[1] * 0.2)  # 20% of image width
        H = self.registration(img1, img2)

        height_img1 = img1.shape[0]
        width_img1  = img1.shape[1]
        width_img2  = img2.shape[1]

        height_panorama = height_img1
        width_panorama  = width_img1 + width_img2

        # Place img1 on the left of a blank canvas, then fade it out near seam
        panorama1 = np.zeros((height_panorama, width_panorama, 3),dtype=np.float32)
        mask1 = self.create_mask(img1, img2, version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1

        # Warp img2 into img1's space using H, then fade it in from seam
        mask2     = self.create_mask(img1, img2, version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2

        # Add both masked images — at the seam, mask1 + mask2 = 1.0 always
        result = panorama1 + panorama2

        # Crop out the black border left by warpPerspective
        rows, cols  = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]

        return final_result


def main(img1_path, img2_path, output_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    scale = 0.3  # try 0.4 or 0.5 if result looks blurry
    img1 = cv2.resize(img1, (0, 0), fx=scale, fy=scale)
    img2 = cv2.resize(img2, (0, 0), fx=scale, fy=scale)
    print(f"Resized to {img1.shape[1]}x{img1.shape[0]}")

    if img1 is None:
        print(f"Could not read image: {img1_path}")
        sys.exit(1)
    if img2 is None:
        print(f"Could not read image: {img2_path}")
        sys.exit(1)

    print(f"Stitching {img1_path} + {img2_path} ...")
    final = Image_Stitching().blending(img1, img2)
    cv2.imwrite(output_path, final)
    print(f"Saved to {output_path}")
    print("Debug match image saved to matching.jpg")


if __name__ == '__main__':
    #  run:  python image_stitching.py
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], OUTPUT_PATH)
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(IMG1_PATH, IMG2_PATH, OUTPUT_PATH)