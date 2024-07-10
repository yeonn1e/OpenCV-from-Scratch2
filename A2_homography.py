import numpy as np
import cv2
import time

cv_cover = cv2.imread('cv_cover.jpg' , cv2.IMREAD_GRAYSCALE)
cv_desk = cv2.imread('cv_desk.png' , cv2.IMREAD_GRAYSCALE)

dia1 = cv2.imread('diamondhead-10.png' , cv2.IMREAD_GRAYSCALE)
dia2 = cv2.imread('diamondhead-11.png' , cv2.IMREAD_GRAYSCALE)

hp_cover = cv2.imread('hp_cover.jpg' , cv2.IMREAD_GRAYSCALE)

# 2-1. feature detection, description, and matching
orb = cv2.ORB_create()

kp_cover = orb.detect(cv_cover, None)
kp_cover, des_cover = orb.compute(cv_cover, kp_cover)

kp_desk = orb.detect(cv_desk, None)
kp_desk, des_desk = orb.compute(cv_desk, kp_desk)

def BFMatcher(des1, des2):
    match_lst = []
    for i in range(len(des1)):
        min = 10000
        for j in range(len(des2)):
            hamming = np.count_nonzero((des1[i]&(2** np.arange(8)).reshape(-1, 1)) != (des2[j]&(2** np.arange(8)).reshape(-1, 1)))
            if min > hamming:
                min, idx = hamming, j

        match_ = cv2.DMatch(i, idx, min)
        match_lst.append(match_)

    match_lst.sort(key=lambda x: x.distance)
    return match_lst

BFMatches = BFMatcher(des_desk, des_cover)
BFMatched_img = cv2.drawMatches(cv_desk, kp_desk, cv_cover, kp_cover,
                                BFMatches[:10], None, flags=2)
cv2.imshow('BFMatched_img',BFMatched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2-2. computing homography with normalization

# srcP, destP: Nx2
# H: 3x3
def wrap(img1, img2):
    return np.where(img2 != 0, img2, img1)

def normalize(mat1, mat2):
    # mean subtraction
    mx1, my1 = np.mean(mat1, axis=0)
    mx2, my2 = np.mean(mat2, axis=0)
    mat1 = np.array([mat1[:,0]-mx1, mat1[:,1]-my1])
    mat2 = np.array([mat2[:,0]-mx2, mat2[:,1]-my2])

    # scaling
    max1, min1 = np.min(mat1), np.max(mat1)
    max2, min2 = np.min(mat2), np.max(mat2)

    M1 = np.array([[1,0,-mx1],
                  [0,1,-my1],
                  [0,0,1]])
    M2 = np.array([[1,0,-mx2],
                  [0,1,-my2],
                  [0,0,1]])
    S1 = np.dot(np.array([[1/(max1-min1),0,0],[0,1/(max1-min1),0],[0,0,1]]),
                np.array([[1,0,-min1],[0,1,-min1],[0,0,1]]))
    S2 = np.dot(np.array([[1/(max2-min2),0,0],[0,1/(max2-min2),0],[0,0,1]]),
                np.array([[1,0,-min2],[0,1,-min2],[0,0,1]]))

    normed_mat1, normed_mat2 = S1.dot(M1), S2.dot(M2)

    return normed_mat1, normed_mat2

def transformation(mat, points):
    # mat: 3x3
    # points: Nx2 입력 좌표 배열

    points = np.hstack([points, np.ones((points.shape[0], 1))])

    transformed = points @ mat.T
    transformed = transformed[:, :2] / transformed[:, 2, np.newaxis]

    return transformed

def compute_homography(srcP, destP):
    # normalize feature points: mean sub, scaling -> srcp, destp
    TS, TD = normalize(srcP, destP)

    # transform -> srcp, destp
    NS = transformation(TS, srcP)
    ND = transformation(TD, destP)

    # calculate A
    A = np.zeros((srcP.shape[0]*2, 9))
    for i in range(srcP.shape[0]):
        sx, sy = NS[i]
        dx, dy = ND[i]
        A[2*i] = [sx, sy, 1, 0, 0, 0, -dx*sx, -dx*sy, -dx]
        A[2*i+1] = [0, 0, 0, sx, sy, 1, -dy*sx, -dy*sy, -dy]

    # SVD
    U, s, Vt = np.linalg.svd(A)

    H = (Vt[-1,:]/Vt[-1,-1]).reshape(3,3)

    invTD = np.linalg.inv(TD)

    XD = np.dot(invTD.dot(H), TS)

    return XD

def Match(des1, des2, N, thresh):
    match_lst = []

    for i in range(len(des1)):
        distances = []
        for j in range(len(des2)):
            hamming = np.count_nonzero((des1[i] & (2**np.arange(8)).reshape(-1, 1)) != (des2[j] & (2**np.arange(8)).reshape(-1, 1)))
            distances.append((hamming, j))

        distances.sort(key=lambda x: x[0])

        closest = distances[0]
        next = distances[1]

        dis_ratio = closest[0] / next[0]

        if dis_ratio < thresh:
            match_ = cv2.DMatch(i, closest[1], closest[0])
            match_lst.append(match_)

    match_lst.sort(key=lambda x: x.distance)
    match_lst = match_lst[:N]
    return match_lst

matches = Match(des_desk, des_cover,15, 0.8) # heuristic

mx = np.array([i.queryIdx for i in matches])
my = np.array([i.trainIdx for i in matches])

desk_pts = np.array([kp_desk[i].pt for i in mx])
cover_pts = np.array([kp_cover[i].pt for i in my])

H = compute_homography(cover_pts, desk_pts)
computed_img = cv2.warpPerspective(cv_cover, H, (cv_desk.shape[1], cv_desk.shape[0]))
cv2.imshow('Homography with normalization',wrap(cv_desk, computed_img)) # Homography with normalization
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2-3. computing homography with RANSAC
# use 2-2. within 3 sec. in or out.

def compute_homography_RANSAC(srcP, destP, th):
    start = time.time()
    selected_inliers = []
    cnt = 0
    iter = 0
    # 1. ransac loop: within 3 sec
    while (time.time()-start < 3) and (iter < 5000):
        iter += 1
        # randomly select a four point correspondences
        random_points = np.random.choice(srcP.shape[0], 4, replace=False)

        # compute H
        H = compute_homography(srcP[random_points], destP[random_points])
        new = transformation(H, srcP)

        # count inliers to the current H
        dis = np.linalg.norm(new - destP, axis=1)
        inliers = np.where(dis <= th)[0]

        # keep H if largest number of inliers
        if len(inliers) > cnt:
            cnt = len(inliers)
            selected_inliers = inliers

    # 2. recompute H using all inliers
    H = compute_homography(srcP[selected_inliers], destP[selected_inliers])

    end = time.time() - start
    print('computational time of RANSAC:', end)
    return H

matches = Match(des_desk, des_cover,15, 0.8) # heuristic

mx = np.array([i.queryIdx for i in matches])
my = np.array([i.trainIdx for i in matches])

desk_pts = np.array([kp_desk[i].pt for i in mx])
cover_pts = np.array([kp_cover[i].pt for i in my])

Hr = compute_homography_RANSAC(cover_pts, desk_pts, 4)
ransac_img = cv2.warpPerspective(cv_cover, Hr, (cv_desk.shape[1], cv_desk.shape[0]))
cv2.imshow('Homography with RANSAC',wrap(cv_desk, ransac_img)) # Homography with RANSAC
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2-4. Image warping -> harry potter
hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
ransac_hp_img = cv2.warpPerspective(hp_cover, Hr, (cv_desk.shape[1], cv_desk.shape[0]))
cv2.imshow('Homography with RANSAC HP',wrap(cv_desk, ransac_hp_img)) # Homography with RANSAC HP
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2-5. Image Stiching
kp_left = orb.detect(dia1, None)
kp_left, des_left = orb.compute(dia1, kp_left)

kp_right = orb.detect(dia2, None)
kp_right, des_right = orb.compute(dia2, kp_right)

stitch_matches = Match(des_left, des_right, 13, 0.8)

mx = np.array([i.queryIdx for i in stitch_matches])
my = np.array([i.trainIdx for i in stitch_matches])

left_pts = np.array([kp_left[i].pt for i in mx])
right_pts = np.array([kp_right[i].pt for i in my])

Hs = compute_homography_RANSAC(right_pts, left_pts, 3)

# a. stich img based on RANSAC
applied_img = cv2.warpPerspective(dia2, Hs, (dia1.shape[1]+int(np.linalg.norm(left_pts[0] - right_pts[0])), dia1.shape[0]))
base_img = np.hstack([dia1, np.zeros((dia1.shape[0],int(np.linalg.norm(left_pts[0] - right_pts[0]))))])
stitched_img = wrap(applied_img, base_img)

cv2.imshow('stitched_img',stitched_img) # stitched_img
cv2.waitKey(0)
cv2.destroyAllWindows()

# b. img blending
# blended img = base img + applied img
for i in range(dia1.shape[1]-200, dia1.shape[1]):
    r = (i-(dia1.shape[1]-200)) / 200
    stitched_img[:,i] = ((1-r)*stitched_img[:,i] + r*applied_img[:,i])

cv2.imshow('blended img',stitched_img) # blended img
cv2.waitKey(0)
cv2.destroyAllWindows()
