import cv2
import numpy as np
from cv2 import aruco
import yaml
import math


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_Perspective = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped_Perspective


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def draw_marker_with_angle(image_points):
    points_im = np.float32([[image_points[0][0][0], image_points[0][0][1]],
                            [image_points[2][0][0], image_points[2][0][1]]])
    if image_points[0][0][0] - image_points[2][0][0] > 0:
        angle = int(
            math.atan((points_im[1][1] - points_im[0][1]) / (points_im[0][0] - points_im[1][0])) * 180 / math.pi)
    else:
        angle = 180 - int(
            math.atan((points_im[1][1] - points_im[0][1]) / (points_im[1][0] - points_im[0][0])) * 180 / math.pi)
    return angle


def find_motion(x, y):
    return (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) > 500


def print_warped(cv_image, warped, id_marker, pts, angle, overlay_1, overlay_2):
    angle = math.radians(angle)
    desired_size = 50
    old_size = warped.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    warped = cv2.resize(warped, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    warped_1 = cv2.copyMakeBorder(warped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    center_marker = (int((pts[0][0] + pts[2][0]) / 2), int((pts[0][1] + pts[2][1]) / 2))
    y1, y2 = int(center_marker[0] - warped_1.shape[0] / 2), int(center_marker[0] + warped_1.shape[0] / 2)
    x1, x2 = int(center_marker[1] - warped_1.shape[1] / 2), int(center_marker[1] + warped_1.shape[1] / 2)
    for c in range(0, 3):
        cv_image[x1:x2, y1:y2, c] = (warped_1[:, :, c] + cv_image[x1:x2, y1:y2, c])
    thickness = 2
    color_circle = (255, 0, 255)
    radius = int(2 * (x2 - x1))
    center = (int(center_marker[0] - math.cos(angle) * (warped_1.shape[0] + warped_1.shape[0] / 2)),
              int(center_marker[1] + math.sin(angle) * (warped_1.shape[1] + warped_1.shape[1] / 2)))
    if overlay_1 > 0:
        if overlay_2 > 0:
            cv2.putText(cv_image, "id: {}".format(overlay_1), (center[0] - 25, center[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.putText(cv_image, "id: {}".format(overlay_2), (center[0] - 25, center[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
        else:
            cv2.putText(cv_image, "id: {}".format(overlay_1), (center[0] - 25, center[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

    cv2.putText(cv_image, "id: {}".format(id_marker), (center[0] - 25, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    if id_marker != -1:
        cv2.circle(cv_image, center, radius, color_circle, thickness)
        cv2.circle(cv_image, center, int(radius // 2), color_circle, thickness)


def print_all_markers(cv_image, height, width, ids, i, id_base_1, id_base_2, id_base_3, overlay_base_1, overlay_base_2,
                      overlay_2_marker, photos, last_motion_photos):
    photos_1 = []
    photos_2 = []
    photos_3 = []
    cv_image[:height, :width] = (155, 0, 0)
    if len(id_base_2) == 0:
        photos_1 = [(warped, h[1], pts, angle) if h[1] == ids[i][0] else h for h in photos]
        photos = photos_1
        for j in range(len(id_base_1)):
            print_warped(cv_image, photos[j][0], photos[j][1], photos[j][2], photos[j][3], 0, 0)
    else:
        for j in range(len(id_base_1 + id_base_2 + id_base_3)):
            if photos[j][1] in id_base_1:
                if photos[j][1] == last_motion_photos[1]:
                    photos_1.append(last_motion_photos)
                else:
                    photos_1.append((photos[j][0], photos[j][1], photos[j][2], photos[j][3]))
            elif photos[j][1] in id_base_2:
                if photos[j][1] == last_motion_photos[1]:
                    photos_2.append(last_motion_photos)
                else:
                    photos_2.append((photos[j][0], photos[j][1], photos[j][2], photos[j][3]))
            elif photos[j][1] in id_base_3:
                if photos[j][1] == last_motion_photos[1]:
                    photos_3.append(last_motion_photos)
                else:
                    photos_3.append((photos[j][0], photos[j][1], photos[j][2], photos[j][3]))
        for j in range(len(id_base_1)):
            if photos_1[j][1] not in overlay_base_1:
                print_warped(cv_image, photos_1[j][0], photos_1[j][1], photos_1[j][2], photos_1[j][3], 0, 0)
        for j in range(len(id_base_2)):
            if photos_2[j][1] not in overlay_base_2:
                for k in range(len(overlay_2_marker)):
                    if photos_2[j][1] == overlay_2_marker[k][1]:
                        print_warped(cv_image, photos_2[j][0], photos_2[j][1], photos_2[j][2], photos_2[j][3],
                                     overlay_2_marker[k][0], 0)
        for j in range(len(id_base_3)):
            for k in range(len(overlay_2_marker)):
                if photos_3[j][1] == overlay_2_marker[k][1]:
                    overlay_2 = [t[0] if t[1] == overlay_2_marker[k][0] else 0 for t in overlay_2_marker]
                    print_warped(cv_image, photos_3[j][0], photos_3[j][1], photos_3[j][2], photos_3[j][3], overlay_2[0],
                                 overlay_2_marker[k][0])
        photos = photos_1 + photos_2 + photos_3
    return photos


filename = 'input files\\input video.mp4'

writer = None
cap = cv2.VideoCapture(filename)
h, w = 1920, 1080

height, width = 1080, 1920
cv_image = np.zeros((height + w, width, 3), np.uint8)
cv_image[:height, :width] = (155, 0, 0)

with open("input files\\calibration_iphone.yaml") as f:
    loadeddict = yaml.load(f)
mtx = loadeddict.get('camera_matrix')
dist = loadeddict.get('dist_coeff')
mtx = np.array(mtx)
dist = np.array(dist)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
arucoParams = aruco.DetectorParameters_create()
length_marker = 4  # in cm
photos = []
max_id_marker = 16
center_of_marker = np.zeros((max_id_marker, 1000, 2), dtype=int)
center_of_marker_last_10 = np.zeros((max_id_marker, 10, 2), dtype=int)
length = 4
marker_frames = np.zeros(max_id_marker, dtype=int)
check_overlay = np.zeros(max_id_marker, dtype=int)
frames = 0
last_motion = 0
draw_after_motion = 0
id_motion = -1
motion = False
overlay = False
id_base = []
id_base_1 = []
id_base_2 = []
id_base_3 = []
overlay_base_1 = []
overlay_base_2 = []
overlay_2_marker = []
counter_overlay = 0
frames_1 = 0
length_of_axis = 10
last_motion_photos = []
axisPoints = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
f = 0
while True:
    (grabbed, image) = cap.read()
    if not grabbed:
        break

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, length, mtx, dist, None, None)
    imaxis = image.copy()

    for i in range(len(ids)):

        imagePoints, jac = cv2.projectPoints(axisPoints, rvec[i], tvec[i], mtx, dist)
        cv2.line(imaxis, (imagePoints[0][0][0], imagePoints[0][0][1]), (imagePoints[2][0][0], imagePoints[2][0][1]),
                 (0, 255, 0), 3)
        angle = draw_marker_with_angle(imagePoints)

        id_marker = ids[i][0]
        marker_frames[id_marker] += 1
        pts = corners[i][0]
        warped = four_point_transform(image, pts)
        center_of_marker[id_marker][marker_frames[id_marker]] = (int((pts[0][0] + pts[2][0]) / 2),
                                                                 int((pts[0][1] + pts[2][1]) / 2))
        if id_marker not in id_base:
            id_base.append(id_marker)
            id_base_1.append(id_marker)
            photos.append((warped, id_marker, pts, angle))
            print_warped(cv_image, photos[-1][0], photos[-1][1], photos[-1][2], photos[-1][3], 0, 0)

        if id_marker in overlay_base_1:
            overlay_base_1.remove(id_marker)
            for j in range(len(overlay_2_marker)):
                if id_marker == overlay_2_marker[j][0]:
                    id_base_2.remove(overlay_2_marker[j][1])
                    id_base_1.append(overlay_2_marker[j][1])
                    overlay_2_marker.remove((id_marker, overlay_2_marker[j][1]))
                    break

        if id_marker in overlay_base_2:
            overlay_base_2.remove(id_marker)
            for j in range(len(overlay_2_marker)):
                if id_marker == overlay_2_marker[j][0]:
                    id_base_3.remove(overlay_2_marker[j][1])
                    id_base_1.append(overlay_2_marker[j][1])
                    overlay_2_marker.remove((id_marker, overlay_2_marker[j][1]))
                    break

        if marker_frames[id_marker] > 2:
            motion = find_motion(center_of_marker[id_marker][marker_frames[id_marker]],
                                 center_of_marker[id_marker][marker_frames[id_marker] - 2])
        if motion:
            motion = False
            last_motion_photos = (warped, id_marker, pts, angle)
            last_motion = marker_frames[id_marker]
            if counter_overlay > 30:
                check_overlay[:] = 0
                counter_overlay = 0
            if id_marker != id_motion:
                print("motion!", "id: ", id_marker)
                counter_overlay = 0
                draw_after_motion = 0
                id_motion = id_marker
                check_overlay[:] = 0
                frames_1 = 0
            photos = print_all_markers(cv_image, height, width, ids, i, id_base_1, id_base_2, id_base_3, overlay_base_1,
                                       overlay_base_2, overlay_2_marker, photos, last_motion_photos)
        if marker_frames[id_motion] - last_motion > 8 and id_marker == id_motion and draw_after_motion < 3:
            last_motion = marker_frames[id_motion]
            draw_after_motion += 1
            photos = print_all_markers(cv_image, height, width, ids, i, id_base_1, id_base_2, id_base_3, overlay_base_1,
                                       overlay_base_2, overlay_2_marker, photos, last_motion_photos)

    if draw_after_motion == 2 and counter_overlay < len(id_base) + 1:
        for i in range(len(id_base)):
            counter_overlay += 1
            check_overlay[int(id_base[i])] = marker_frames[int(id_base[i])]
    if check_overlay[id_base[0]] != 0:
        counter_overlay += 1

    for i in range(len(id_base_1)):
        '''
        detect overlayed markers on 1 level
        '''
        if draw_after_motion > 2 and frames_1 == 0 and\
                marker_frames[int(id_base_1[i])] - check_overlay[int(id_base_1[i])] == 0 and\
                counter_overlay > len(id_base) + 2 and id_base_1[i] not in overlay_base_1:
            frames_1 += 1
            print("overlay! id_marker: ", id_base_1[i])
            id_base_2.append(id_motion)
            overlay_base_1.append(id_base_1[i])
            overlay_2_marker.append((id_base_1[i], id_motion))
            id_base_1.remove(id_motion)
            photos = print_all_markers(cv_image, height, width, ids, i, id_base_1, id_base_2, id_base_3, overlay_base_1,
                                       overlay_base_2, overlay_2_marker, photos, last_motion_photos)
            break
    for i in range(len(id_base_2)):
        '''
        detect overlayed markers on 2 level
        '''
        if draw_after_motion > 2 and frames_1 == 0 \
                and marker_frames[int(id_base_2[i])] - check_overlay[int(id_base_2[i])] == 0 \
                and counter_overlay > len(id_base) + 2 and id_base_2[i] not in overlay_base_2:
            frames_1 += 1
            print("overlay! id_marker: ", id_base_2[i])
            id_base_3.append(id_motion)
            overlay_base_2.append(id_base_2[i])
            overlay_2_marker.append((id_base_2[i], id_motion))
            id_base_1.remove(id_motion)
            photos = print_all_markers(cv_image, height, width, ids, i, id_base_1, id_base_2, id_base_3, overlay_base_1,
                                       overlay_base_2, overlay_2_marker, photos, last_motion_photos)
            break

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output\\output video.avi", fourcc, 15,
                                 (cv_image.shape[1], cv_image.shape[0]), True)
    cv2.putText(cv_image, "id_base_1: {}".format(id_base_1), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 4)
    cv2.putText(cv_image, "id_base_2: {}".format(id_base_2), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 4)
    cv2.putText(cv_image, "id_base_3: {}".format(id_base_3), (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 4)
    cv_image[height:, :h] = imaxis[:, :]
    writer.write(cv_image)
    frames += 1
writer.release()
cap.release()
cv2.destroyAllWindows()
