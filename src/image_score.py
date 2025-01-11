import sys
import cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np
# from src.ppocr.utils.logging import get_logger
# logger = get_logger()

# define the map from indication to the radian 
indication_map = {
    12: (math.pi / 6) * 3, 11: (math.pi / 6) * 4, 10: (math.pi / 6) * 5,
    9: (math.pi / 6) * 6, 8: (math.pi / 6) * 7, 7: (math.pi / 6) * 8,
    6: (math.pi / 6) * 9, 5: (math.pi / 6) * 10, 4: (math.pi / 6) * 11,
    3: 0, 2: math.pi / 6, 1: (math.pi / 6) * 2, 0: (math.pi / 6) * 3
}


sample_diff_details = [dict({})]
image_full_path = ""


def is_noise_point(img, row, col):
    scan_range = 0.1
    height = img.shape[0]
    width = img.shape[1]
    point_count = 0
    # axis = 0
    scan_low_bound_axis0 = max(0, int(row - 50))
    scan_high_bound_axis0 = min(int(row + 50), height)
    # axis = 1
    scan_low_bound_axis1 = max(0, int(col - 50))
    scan_high_bound_axis1 = min(int(col + 50), width)
    for axis0 in range(scan_low_bound_axis0, scan_high_bound_axis0):
        for axis1 in range(scan_low_bound_axis1, scan_high_bound_axis1):
            if img[axis0][axis1] == 0:
                point_count += 1
            if point_count >= 100:
                return False
    return True


def get_drawing_range(img):
    """获取图片"""
    height = img.shape[0]
    width = img.shape[1]
    hit_points = []
    # axis = 0
    for row in range(height - 1):
        # axis = 1
        for col in range(width - 1):
            if img[row][col] != 0:
                continue
            if is_noise_point(img, row, col):
                continue
            hit_points.append((row, col))
    points_axis0 = []
    points_axis1 = []
    for point in hit_points:
        points_axis0.append(point[0])
        points_axis1.append(point[1])

    bottom = 0 if len(points_axis0) == 0 else min(points_axis0)
    top = 0 if len(points_axis0) == 0 else max(points_axis0)
    left = 0 if len(points_axis1) == 0 else min(points_axis1)
    right = 0 if len(points_axis1) == 0 else max(points_axis1)
    # return (min(points_axis0), max(points_axis0), min(points_axis1), max(points_axis1))
    return (bottom, top, left, right)



def get_circle_radius(img, bottom, top, left, right, outter=False):
    height = img.shape[0]
    width = img.shape[1]
    # print("img range : left = {}, right = {}, bottom = {}, top = {}".format(left, right, bottom, top))
    # draw a cross line
    # cv.line(img, (left, bottom), (right, top), (0, 255, 0), 2, 8, 0)

    # DEBUG: draw a rectangle
    # cv.rectangle(img, (left, bottom), (right, top), (0, 255, 0), 2)

    origin_x, origin_y = left + (right - left) / 2, bottom + (top - bottom) / 2
    radius = (top - bottom) / 2 if ((top - bottom) / 2) < ((right - left) / 2) else (right - left) / 2
    # get the outter circle
    if outter:
        return radius, origin_x, origin_y

    radius = min(min((origin_x - left), (right - origin_x)), radius)
    radius = min(min((origin_y - bottom), (top - origin_y)), radius)

    # draw standard circle's origin point
    cv.circle(img, (np.int32(origin_x), np.int32(origin_y)), 10, (0, 255, 0), -1, 8, 0)
    # cv.imshow("yuanxin", img)
    return radius, origin_x, origin_y



def get_drawing_point_coord(img, coord_v, range_lo, range_hi, axis=1):
    if axis == 1:
        coord_lo = range_hi
        coord_hi = range_lo
        coord_mid = int((range_lo + range_hi) / 2)
        found_lo = False
        found_hi = False
        hit_y_points = [coord_mid]
        # print("LO = {}, HI = {}".format(range_lo, range_hi))
        for y in range(range_lo, range_hi):
            if img[y][coord_v] != 0:
                continue
            hit_y_points.append(y)
        return (min(hit_y_points), max(hit_y_points))
    elif axis == 0:
        coord_left = range_hi
        coord_right = range_lo
        coord_mid = int((range_lo + range_hi) / 2)
        found_left = False
        found_right = False
        hit_x_points = [coord_mid]
        # print("LEFT = {}, RIGHT = {}".format(range_lo, range_hi))
        for x in range(range_lo, range_hi):
            if img[coord_v][x] != 0:
                continue
            hit_x_points.append(x)
        return (min(hit_x_points), max(hit_x_points))


def get_inscribed_circle_coord(origin_x, origin_y, r, v, axis=1):
    if axis == 1:
        # (x - r) * (x - r) + (y - r) * (y - r) = r * r
        # print("origin_x = {}, origin_y = {}, r = {}, x = {}".format(origin_x, origin_y, r, x))
        penalty_rate = 1 # abs(x - origin_x) / 200
        if r * r < ((v - origin_x) * (v - origin_x)):
            # print("miss drawing range, return ({}, {})".format(origin_y * penalty_rate, origin_y * penalty_rate))
            return (origin_y * penalty_rate, origin_y * penalty_rate)
        y1 = origin_y - math.sqrt(r * r - ((v - origin_x) * (v - origin_x)))
        y2 = origin_y + math.sqrt(r * r - ((v - origin_x) * (v - origin_x)))
        return (y1, y2)
    elif axis == 0:
        penalty_rate = 1 # abs(x - origin_x) / 200
        if r * r < ((v - origin_y) * (v - origin_y)):
            # print("miss drawing range, return ({}, {})".format(origin_x * penalty_rate, origin_x * penalty_rate))
            return (origin_x * penalty_rate, origin_x * penalty_rate)
        x1 = origin_x - math.sqrt(r * r - ((v - origin_y) * (v - origin_y)))
        x2 = origin_x + math.sqrt(r * r - ((v - origin_y) * (v - origin_y)))
        return (x1, x2)


def sample_and_diff(img, radius, origin_x, origin_y, bottom, top, left, right, image_full_path=""):
    power2_delta = 0
    area_diff = 0
    up_down_symmetry_deviation = 0
    bilateral_symmetry_deviation = 0
    for x in range(left, right):
        (coord_lo, coord_hi) = get_drawing_point_coord(img, x, bottom, top)
        (y1, y2) = get_inscribed_circle_coord(origin_x, origin_y, radius, x)
        # cv.circle(img, (np.int32(x), np.int32(y1)), 1, (0, 255, 0), 2, 8, 0)
        # cv.circle(img, (np.int32(x), np.int32(y2)), 1, (0, 255, 0), 2, 8, 0)
        # print("x = {}, coord_lo = {}, y1 = {}, --|-- coord_hi = {}, y2 = {}".format(x, coord_lo, y1, coord_hi, y2))
        coord_mid = (coord_hi + coord_lo) / 2
        # only accumulate area diff in x asix computing logic
        area_diff += ((y1 - coord_lo) * (y1 - coord_lo) + (y2 - coord_hi) * (y2 - coord_hi))
        up_down_symmetry_deviation += (coord_mid - origin_y) * (coord_mid - origin_y)
    for y in range(bottom, top):
        (coord_left, coord_right) = get_drawing_point_coord(img, y, left, right, axis=0)
        (x1, x2) = get_inscribed_circle_coord(origin_x, origin_y, radius, y, axis=0)
        # cv.circle(img, (np.int32(x), np.int32(y1)), 1, (0, 255, 0), 2, 8, 0)
        # cv.circle(img, (np.int32(x), np.int32(y2)), 1, (0, 255, 0), 2, 8, 0)
        # print("y = {}, coord_left = {}, x1 = {}, --|-- coord_right = {}, x2 = {}".format(y, coord_left, x1, coord_right, x2))
        coord_mid = (coord_left + coord_right) / 2
        bilateral_symmetry_deviation += (coord_mid - origin_x) * (coord_mid - origin_x)
    area_diff = area_diff / (radius * radius) / (right - left) if right > left else area_diff
    up_down_symmetry_deviation = up_down_symmetry_deviation / (radius * radius) / (right - left) if right > left else up_down_symmetry_deviation
    bilateral_symmetry_deviation = bilateral_symmetry_deviation / (radius * radius) / (top - bottom) if top > bottom else bilateral_symmetry_deviation
    power2_delta = (area_diff + up_down_symmetry_deviation + bilateral_symmetry_deviation)
    diff_dict = {"image" : image_full_path, "area_diff" : area_diff, "up_down_symmetry_deviation" : up_down_symmetry_deviation, "bilateral_symmetry_deviation" : bilateral_symmetry_deviation, "power2_delta": power2_delta}
    sample_diff_details.append(diff_dict)
    # logger.debug("sample_and_diff = {}".format(power2_delta))
    return power2_delta


def preprocess_img(img_dir):
    # preprocess image
    raw = cv.imread(img_dir)
    img = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
    # img = cv.medianBlur(img, 3)  # 模糊降噪
    img = cv.GaussianBlur(img, (5, 5), 5)  # 模糊降噪
    img = cv.GaussianBlur(img, (5, 5), 5)  # 模糊降噪
    img = cv.GaussianBlur(img, (5, 5), 5)  # 模糊降噪
    # canny = cv.Canny(gray, 5, 100)
    # ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    height, width = img.shape[0], img.shape[1]

    max_side = max(width, height)
    if max_side > 300:
        ratio = 300 / max_side
        width, height = int(width * ratio), int(height * ratio)
        img = cv.resize(img, (width, height))


    eer = 0.02  # edge erase ratio
    for row in range(height - 1):
        for col in range(width - 1):
            if (col < width * eer) or (col > width * (1 - eer)) or (row > height * (1 - eer)) or (row < height * eer):
                img[row][col] = 255
                # print("img[{}][{}] = {}".format(col, row, img[col][row]))
    return img


def draw_line(img, origin_x, origin_y, radius, indication):
    dd_x = (radius / 2) * math.cos(indication_map[indication])
    dd_y = (radius / 2) * math.sin(indication_map[indication])
    end_p_x = np.int32(origin_x + dd_x)
    end_p_y = np.int32(origin_y - dd_y)
    cv.line(img, (np.int32(origin_x), np.int32(origin_y)),
            (end_p_x, end_p_y), (0, 255, 0), 2, 8, 0)


# y = x * sin(angle)
def find_indication(img, origin_x, origin_y, radius, indication):
    hit_ratio = 0.4
    hit_count = 0
    error_tolerance = np.int32(radius / 3 / 2)
    hit_points = []
    hit_threshold = np.int32(radius * hit_ratio)
    for d in range(0, np.int32(radius * 0.5)):
        d_x = np.int32(d * math.cos(indication_map[indication]))
        d_y = np.int32(d * math.sin(indication_map[indication]))
        p_d_x = np.int32(origin_x + d_x)
        p_d_y = np.int32(origin_y - d_y)
        find_point = False
        if (indication == 12) or (indication == 6):
            for i in range(p_d_x - error_tolerance, p_d_x + error_tolerance):
                # cv.circle(img, (np.int32(i), np.int32(p_d_y)), 1, (0, 255, 0), 2, 8, 0)
                if (img[p_d_y][i] == 0) and (not find_point):
                    # print("find hit point img[{}][{}] = {}".format(j, p_d_x, img[j][p_d_x]))
                    hit_count += 1
                    find_point = True
                    hit_points.append((i, p_d_y))
        else:
            for j in range(p_d_y - error_tolerance, p_d_y + error_tolerance):
                # cv.circle(img, (np.int32(p_d_x), np.int32(j)), 1, (0, 255, 0), 2, 8, 0)
                if (img[j][p_d_x] == 0) and (not find_point):
                    # print("find hit point img[{}][{}] = {}".format(j, p_d_x, img[j][p_d_x]))
                    hit_count += 1
                    find_point = True
                    hit_points.append((p_d_x, j))
                    break

    if hit_count > hit_threshold:
        # cv.line(img, hit_points[0], hit_points[hit_threshold], (0, 255, 0), 2, 8, 0)
        slope = abs(hit_points[hit_threshold][1] - hit_points[0][1]) / math.sqrt(
            (hit_points[hit_threshold][1] - hit_points[0][1]) * (hit_points[hit_threshold][1] - hit_points[0][1]) +
             (hit_points[hit_threshold][0] - hit_points[0][0]) * (hit_points[hit_threshold][0] - hit_points[0][0]))

        # print("hit_points: head {}, tail {}".format(hit_points[hit_threshold], hit_points[0]))
        # print("hand slope = {} vs indication {}'s standard value {}".format(slope, indication, math.sin(indication_map[indication])))
        if abs(abs(slope) - abs(math.sin(indication_map[indication]))) < 0.13:
            # cv.line(img, hit_points[0], hit_points[hit_threshold], (0, 255, 0), 2, 8, 0)
            return True
    
    # draw standard hands
    # draw_line(img, origin_x, origin_y, radius, indication)

    return False


def find_hands(img, origin_x, origin_y, radius):
    hand_count = 0
    for ind in range(1, 12 + 1):
        is_hand_here = find_indication(img, origin_x, origin_y, radius, ind)
        if is_hand_here:
            hand_count += 1
        if hand_count == 2:
            break
    return hand_count


def get_hand_len(img, origin_x, origin_y, radius, indication):
    hit_ratio = 0.8
    hit_count = 0
    # the tail of hand line
    hand_start_x = origin_x
    hand_start_y = origin_y
    hand_end_x = origin_x
    hand_end_y = origin_y
    error_tolerance = np.int32(radius / 3 / 2)
    # print("indication = {}".format(indication))
    for d in range(0, np.int32(radius / 2)):
        d_x = np.int32(d * math.cos(indication_map[indication]))
        d_y = np.int32(d * math.sin(indication_map[indication]))
        p_d_x = np.int32(origin_x + d_x)
        p_d_y = np.int32(origin_y - d_y)
        find_point = False
        if (indication == 12) or (indication == 6):
            for i in range(p_d_x - error_tolerance, p_d_x + error_tolerance):
                # cv.circle(img, (np.int32(i), np.int32(p_d_y)), 1, (0, 255, 0), 2, 8, 0)
                if (img[p_d_y][i] == 0) and (not find_point):
                    hit_count += 1
                    find_point = True
                    hand_end_x = i
                    hand_end_y = p_d_y
        else:
            for j in range(p_d_y - error_tolerance, p_d_y + error_tolerance):
                # cv.circle(img, (np.int32(p_d_x), np.int32(j)), 1, (0, 255, 0), 2, 8, 0)
                if (img[j][p_d_x] == 0) and (not find_point):
                    # print("find hit point img[{}][{}] = {}".format(j, p_d_x, img[j][p_d_x]))
                    hit_count += 1
                    find_point = True
                    hand_end_x = p_d_x
                    hand_end_y = j

    # cv.line(img, (np.int32(hand_start_x), np.int32(hand_start_y)),
    #         (hand_end_x, hand_end_y), (0, 255, 0), 2, 8, 0)
    hand_len = math.sqrt(
        (hand_end_x - hand_start_x) * (hand_end_x - hand_start_x) + (hand_end_y - hand_start_y) * (hand_end_y - hand_start_y))
    return hand_len


def get_clock_integrity_score(diff):
    clock_score = 0
    if diff < 0.02:
        clock_score = 4
    elif diff < 0.05:
        clock_score = 3 + (diff - 0.02) / (0.05 - 0.02)
    elif diff < 0.1:
        clock_score = 2 + (diff - 0.05) / (0.1 - 0.05)
    elif diff < 0.2:
        clock_score = 1 + (diff - 0.1) / (0.2 - 0.1)
    elif diff < 1:
        clock_score = 0 + (diff - 0.2) / (1 - 0.2)
    else:
        clock_score = 0
    return round(clock_score, 2)


def score_circle_and_hands(img_dir, hour, minute):
    """
    coordinate image
    """
    print("computing score of {}, with hour = {}, minute = {}".format(img_dir, hour, minute))
    img = preprocess_img(img_dir)
    bottom, top, left, right = get_drawing_range(img)
    (radius, origin_x, origin_y) = get_circle_radius(img, bottom, top, left, right)
    # logger.debug("radius = {}, origin_x = {}, origin_y = {}".format(
    #     radius, origin_x, origin_y))

    image_full_path = img_dir.replace("/", "_")
    diff = sample_and_diff(img, radius, origin_x, origin_y, bottom, top, left, right,image_full_path)
    score_circle = get_clock_integrity_score(diff)
    print("diff = {}, clock score_circle = {}".format(diff, score_circle))

    # DEBUG: draw standard circle
    # cv.circle(img, (np.int32(origin_x), np.int32(origin_y)),
    #           np.int32(radius), (0, 255, 0), 2, 8, 0)
    # cv.imwrite("./tmp/" + img_dir.replace("/", "_") + "_csc_" + str(diff) + ".png", img)
    # return

    # logger.info("find {} hands in the clock face, get {} scores".format(hands, hands))

    # convert hour and minute numbers into clock indication
    hour_indication = hour + 1 if minute > 30 else hour
    minute_indication = 12 if minute == 0 else round(minute / 5)
    is_hour_hand_here = False
    is_minute_hand_here = False

    # if there are two hands
    # hands = find_hands(img, origin_x, origin_y, radius)
    hands = 0

    # is_hour_hand_here = find_indication(img, origin_x, origin_y, radius, hour_indication)
    # is_minute_hand_here = find_indication(img, origin_x, origin_y, radius, minute_indication)


    for oxx in range(np.int32(origin_x - radius * 0.2), np.int32(origin_x + radius * 0.2)):
        for oyy in range(np.int32(origin_y - radius * 0.2), np.int32(origin_y + radius * 0.2)):
            is_hour_hand_here = find_indication(
                img, oxx, oyy, radius, hour_indication)
            if is_hour_hand_here:
                break
        if is_hour_hand_here:
            break

    for oxx in range(np.int32(origin_x - radius * 0.2), np.int32(origin_x + radius * 0.2)):
        for oyy in range(np.int32(origin_y - radius * 0.2), np.int32(origin_y + radius * 0.2)):
            is_minute_hand_here = find_indication(
                img, oxx, oyy, radius, minute_indication)
            if is_minute_hand_here:
                break
        if is_minute_hand_here:
            break

    print("is_hour_hand_here ? {}".format("YES" if is_hour_hand_here else "NO"))
    print("is_minute_hand_here ? {}".format("YES" if is_minute_hand_here else "NO"))

    if is_hour_hand_here:
        hands += 1
    if is_minute_hand_here:
        hands += 1

    score_hands = 40 * hands  # 40 score for each hand
    # if minute hand is longer than hour hand, add 20 score
    if is_hour_hand_here and is_minute_hand_here:
        hour_hand_len = get_hand_len(
            img, origin_x, origin_y, radius, hour_indication)
        minute_hand_len = get_hand_len(
            img, origin_x, origin_y, radius, minute_indication)
        print("hour_hand_len = {}, minute_hand_len = {}".format(
            hour_hand_len, minute_hand_len))
        if minute_hand_len > hour_hand_len:
            score_hands += 20

    # draw standard circle
    cv.circle(img, (np.int32(origin_x), np.int32(origin_y)),
              np.int32(radius), (0, 255, 0), 2, 8, 0)
    cv.imwrite("./tmp/" + img_dir.replace("/", "_") + "_debug.png", img)

    return score_circle, score_hands


if __name__ == "__main__":
    img_dir = sys.argv[1]
    hour = int(sys.argv[2])
    minute = int(sys.argv[3])
    # test input time
    score_circle_and_hands(img_dir, hour, minute)
    with open("./tmp/circle_diff_details.json", "w") as fout:
        json.dump(sample_diff_details, fout, indent=2)


# if __name__ == "__main__":
#     dir = sys.argv[1]
#     # test input time

#     for parent, dirnames, filenames in os.walk(dir):
#         for img_name in filenames:
#             img_path = os.path.join(parent, img_name)
#             print("process {}".format(img_path))
#             score_circle_and_hands(img_path, 2, 30)
#     with open("./tmp/circle_diff_details.json", "w") as fout:
#         json.dump(sample_diff_details, fout, indent=2)

