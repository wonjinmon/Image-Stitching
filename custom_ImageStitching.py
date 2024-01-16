'''
    23.10.16. Seokju Lee

    Original code from:
    https://gist.github.com/tigercosmos/90a5664a3b698dc9a4c72bc0fcbd21f4
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import pdb
import matplotlib
matplotlib.use('tkagg')


# Fix the size of visualization
plt.rcParams['figure.figsize'] = [15, 15]


# Read image and convert them to gray!
def read_image(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb


left_gray, left_origin, left_rgb = read_image('data/3.jpg')
right_gray, right_origin, right_rgb = read_image('data/4.jpg')

# 합친 이미지 사이즈 맞춰주기

# print(left_gray.shape, right_gray.shape)

def SIFT(img):
    siftDetector = cv2.SIFT_create()
    kp, des = siftDetector.detectAndCompute(img, None)  # kp: keypoints, des: descriptor
    print(len(kp), len(des))
    return kp, des


def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img


# Better result when using gray. Why?
kp_left, des_left = SIFT(left_gray)
kp_right, des_right = SIFT(right_gray)

kp_left_img = plot_sift(left_gray, left_rgb, kp_left)
kp_right_img = plot_sift(right_gray, right_rgb, kp_right)

total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
'''
    Q. What is the meaning of the "axis" option?
    A.
    The axis along which the arrays will be joined.  If axis is None,
    arrays are flattened before use.  Default is 0.
    기준이 되는 축을 설정함. 0 or None이면 사진concat방향이 row방향 증가 / 1이면 column방향으로 증가.
'''

plt.imshow(total_kp)
plt.ion()
plt.show()
pdb.set_trace()
'''
    Q. Please check the plot. How many keypoints are extracted for each image?
    A.
    kp_left_img : 8167
    kp_right_img: 8411

    Q. Please check whether the number of keypoints and the size of the descriptor are the same.
    A.
    kp_left_img : 8167 / 8167
    kp_right_img: 8411 / 8411
    서로 같다.
'''

print("Completed SIFT feature extraction!")


def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    '''
        Q. We used "bf.match()" in "SIFT_Matching.py". Please discuss the differences between "bf.match()"" and "bf.knnMatch()".
        (reference: https://leechamin.tistory.com/330)
        A.
        가장 일치하는 K개를 반환. 파라미터로 정해지는 K. 8167개
    '''
    # print(len(matches))
    # print(type(matches))

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    '''
        Q. Please discuss the mechanism how to filter out noisy matches.
        A.
        임계값을 설정해서 구한 n에서의 descriptor사이의 거리(필터)보다 m에서의 descriptor사이의 거리가 작으면 통과(필터 통과)
    '''
    # print(len(good))

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    # print(len(matches))
    return matches


matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)
print("Completed SIFT matching!", matches)
pdb.set_trace()
'''
    Q. Please check how many (ratio) keypoints are filtered out after the matching.
    A.
    8167 -> 1196개로 감소했음.
'''


def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8'))  # RGB is integer type

    ax.plot(matches[:, 0], matches[:, 1], 'xr')  # 
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')

    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.ion()
    plt.show()


total_img = np.concatenate((left_rgb, right_rgb), axis=1)
plot_matches(matches, total_img)  # Good mathces
pdb.set_trace()
'''
    Q. Please specify the information that "matches" contains.
    A.
    두 이미지 사이에서 찾은 특정점들 중 매칭이 완료된 점들 (?)
'''


def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    '''
        Q. Please explain the mechanism of "Singular Value Decomposition". Why do we need it?
        (reference: https://darkpgmr.tistory.com/108)
        A.
        SVD :  A = U Σ V^T의 행렬곱으로 분해할 수 있다는 decomposition중 하나이다.
        {A= (M,N), U, V = Orthogonal Matrix, Σ = Digonal Matrix}

        Least Square Method를 통해 해를 구하고자 할 때, 이 때 역행렬을 pseudo inverse대신 SVD를 통해 구할 수도 있다.
        (over-determined case)

        homogeneous 연립선형방정식의 해를 SVD를 통해 구할 수 있음.

        SFM에서 SVD & Newton's Method 이용 -> Camera Motion과 Scene Structure를 구할 수 있음.
    '''
    # print(V)
    H = V[-1].reshape(3, 3)

    H = H/H[2, 2]  # standardize to let w*H[2,2] = 1
    '''
        Q. Why do we need to normalize it?
        A.
        그 Projective Matrix(Homography) 형태로 맞춰주는 과정. H = (3,3)
        자유도를 8로 만들어서 매칭되는 점 4개로 변환 가능하게 하는 ㅇㅇ
    '''
    return H


def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx]
    return np.array(point)


def get_error(points, H):
    '''
        Q. Please describe how the error is defined.
        A.
        all_p1(왼쪽 이미지 좌표)에 H(homography)를 곱하고 인덱싱/슬라이싱한 estimate_p2와
        all_p2(오른쪽 이미지 좌표)의 거리를 유클리드거리를 계산해서
        예측한(변환한) 값과 실제 점 사이의 error를 구함.
    '''
    # points.shape = 1196,4
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)  # 왼쪽(첫 번째) 이미지 좌표
    all_p2 = points[:, 2:4]  # 오른쪽(두 번째) 이미지 좌표
    estimate_p2 = np.zeros((num_points, 2))

    for i in range(num_points):
        # print(H.shape, all_p1.shape)
        temp = np.dot(H, all_p1[i])  # 행렬곱
        # print(temp)
        estimate_p2[i] = (temp/temp[2])[0:2]  # set index 2 to 1 and slice the index 0, 1
    # Compute error
    # estimate_p2.shape = 1196,2
    errors = np.linalg.norm(all_p2 - estimate_p2, axis=1) ** 2

    return errors


### RANSAC Operation ###
def ransac(matches, threshold, iters=2000):
    num_best_inliers = 0

    for _ in range(iters):
        points = random_point(matches)
        H = homography(points)  # candidate

        # Avoid dividing by zero. Why?
        if np.linalg.matrix_rank(H) < 3:
            continue

        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()

    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H


inliers, H = ransac(matches, 0.5, 500)
print("Completed RANSAC operation!")

plot_matches(inliers, total_img)  # show inliers matches
pdb.set_trace()
'''
    Q. Please understand the RANSAC operation.
    -> ㅇㅇ

    Q. Change the number of iterations. What happens? Please discuss the speed and accuracy.
    A.
    seed(77)로 실험
    iter=100: 378/1196, iter=3000: 455/1196
    iter가 길수록 시간이 오래걸림.
    iter가 많을 수록 정확도가 높아지는 것은 아님.
    RANSAC이 랜덤base이기 때문.

    Q. Please check how many (ratio) matched points are filtered out after the RANSAC operation.
    A.
    1196개 -> 약 4~500개.
'''


def stitch_img(left, right, H):
    '''
        Q. In this function, there are two "cv2.warpPerspective()" operations. Please discuss why.
        A.
        이미지 스티칭을 하기위해서 이미지를 변환하는거임.
        평평한 상태의 이미지 2개로는 서로 붙이기 어려우니 perspective transfrom을 가해서 이미지를 변환을 줌.
    '''
    print("stiching image ...")

    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   

    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)

    '''
        Q. Why do we need to multiply "translation_mat" and "H"?
        A.
        H(homography)에 아핀변환을 적용해주기 위함. "Translation"
    '''

    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    print(height_new, width_new, size)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape

    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)

    print(height_new, width_new, size)

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)

    black = np.zeros(3)  # Black pixel.

    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):    # Q. What does "tqdm" do?
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]

            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass

    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image


### Operate image stitching ###
stitched_img = stitch_img(left_rgb, right_rgb, H)
plt.imshow(stitched_img)
# plt.imsave('./data/ALL.jpg', stitched_img)
plt.ion()
plt.show()
pdb.set_trace()
