from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import matplotlib.patches as patches
import glob


def preprocess_v1(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode


def find_tip_v1(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])


def arrowhead_detection_v1(img_path):
    img = cv2.imread(str(img_path))
    contours, hierarchy = cv2.findContours(preprocess_v1(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Konwersja kolorów z BGR na RGB

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        if 6 > sides > 3 and sides + 2 == len(approx):
            arrow_tip = find_tip_v1(approx[:, 0, :], hull.squeeze())
            if arrow_tip:
                cnt_transformed = cnt[:, 0, :]
                polygon = patches.Polygon(cnt_transformed, edgecolor=(0, 1, 0), facecolor="none")
                ax.add_patch(polygon)

                arrow_tip_x, arrow_tip_y = arrow_tip
                circle = patches.Circle((arrow_tip_x, arrow_tip_y), radius=3, edgecolor=(1, 0, 0), facecolor=(1, 0, 0))
                ax.add_patch(circle)

    plt.show()


def components_detection_v2(img_path):
    img = cv2.imread(str(img_path),0)

    # _,img = cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    labels, stats = cv2.connectedComponentsWithStats(img, 8)[1:3]

    for label in np.unique(labels)[1:]:

        arrow = labels==label
        indices = np.transpose(np.nonzero(arrow)) #y,x
        dist = distance.cdist(indices, indices, 'euclidean')
        far_points_index = np.unravel_index(np.argmax(dist), dist.shape) #y,x
        far_point_1 = indices[far_points_index[0],:] # y,x
        far_point_2 = indices[far_points_index[1],:] # y,x
        ### Slope
        arrow_slope = (far_point_2[0]-far_point_1[0])/(far_point_2[1]-far_point_1[1])
        arrow_angle = math.degrees(math.atan(arrow_slope))
        ### Length
        arrow_length = distance.cdist(far_point_1.reshape(1,2), far_point_2.reshape(1,2), 'euclidean')[0][0]
        ### Thickness
        x = np.linspace(far_point_1[1], far_point_2[1], 20)
        y = np.linspace(far_point_1[0], far_point_2[0], 20)
        line = np.array([[yy,xx] for yy,xx in zip(y,x)])
        thickness_dist = np.amin(distance.cdist(line, indices, 'euclidean'),axis=0).flatten()
        n, bins, patches = plt.hist(thickness_dist,bins=150)
        thickness = 2*bins[np.argmax(n)]

        print(f"Thickness: {thickness}")
        print(f"Angle: {arrow_angle}")
        print(f"Length: {arrow_length}\n")
        plt.figure()
        plt.imshow(arrow,cmap='gray')
        plt.scatter(far_point_1[1],far_point_1[0],c='r',s=10)
        plt.scatter(far_point_2[1],far_point_2[0],c='r',s=10)
        plt.scatter(line[:,1],line[:,0],c='b',s=10)
        plt.show()


def rotateBound_v7(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def arrow_v7(img_path):

    inputImage = cv2.imread(str(img_path))

    # Grayscale conversion:
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    grayscaleImage = 255 - grayscaleImage

    # Find the big contours/blobs on the binary image:
    contours, hierarchy = cv2.findContours(grayscaleImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Process each contour 1-1:
    for i, c in enumerate(contours):
        # Approximate the contour to a polygon:
        contoursPoly = cv2.approxPolyDP(c, 3, True)

        # Convert the polygon to a bounding rectangle:
        boundRect = cv2.boundingRect(contoursPoly)

        # Get the bounding rect's data:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]

        # Get the rect's area:
        rectArea = rectWidth * rectHeight

        minBlobArea = 100
        # Check if blob is above min area:
        if rectArea > minBlobArea:
            # Crop the roi:
            croppedImg = grayscaleImage[rectY:rectY + rectHeight, rectX:rectX + rectWidth]

            # Extend the borders for the skeleton:
            borderSize = 5
            croppedImg = cv2.copyMakeBorder(croppedImg, borderSize, borderSize, borderSize, borderSize,
                                            cv2.BORDER_CONSTANT)

            # Store a deep copy of the crop for results:
            grayscaleImageCopy = cv2.cvtColor(croppedImg, cv2.COLOR_GRAY2BGR)

            # Compute the skeleton:
            skeleton = cv2.ximgproc.thinning(croppedImg, None, 1)
            # Threshold the image so that white pixels get a value of 0 and
            # black pixels a value of 10:
            _, binaryImage = cv2.threshold(skeleton, 128, 10, cv2.THRESH_BINARY)

            # Set the end-points kernel:
            h = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]])

            # Convolve the image with the kernel:
            imgFiltered = cv2.filter2D(binaryImage, -1, h)

            # Extract only the end-points pixels, those with
            # an intensity value of 110:
            binaryImage = np.where(imgFiltered == 110, 255, 0)
            # The above operation converted the image to 32-bit float,
            # convert back to 8-bit uint
            binaryImage = binaryImage.astype(np.uint8)
            # Find the X, Y location of all the end-points
            # pixels:
            Y, X = binaryImage.nonzero()

            # Check if I got points on my arrays:
            if len(X) > 0 or len(Y) > 0:
                # Reshape the arrays for K-means
                Y = Y.reshape(-1, 1)
                X = X.reshape(-1, 1)
                Z = np.hstack((X, Y))

                # K-means operates on 32-bit float data:
                floatPoints = np.float32(Z)

                # Set the convergence criteria and call K-means:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, label, center = cv2.kmeans(floatPoints, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                # Set the cluster count, find the points belonging
                # to cluster 0 and cluster 1:
                cluster1Count = np.count_nonzero(label)
                cluster0Count = np.shape(label)[0] - cluster1Count

                # Look for the cluster of max number of points
                # That cluster will be the tip of the arrow:
                maxCluster = 0
                if cluster1Count > cluster0Count:
                    maxCluster = 1

                # Check out the centers of each cluster:
                matRows, matCols = center.shape
                # We need at least 2 points for this operation:
                if matCols >= 2:
                    # Store the ordered end-points here:
                    orderedPoints = [None] * 2
                    # Let's identify and draw the two end-points
                    # of the arrow:
                    for b in range(matRows):
                        # Get cluster center:
                        pointX = int(center[b][0])
                        pointY = int(center[b][1])
                        # Get the "tip"
                        if b == maxCluster:
                            color = (0, 0, 255)
                            orderedPoints[0] = (pointX, pointY)
                        # Get the "tail"
                        else:
                            color = (255, 0, 0)
                            orderedPoints[1] = (pointX, pointY)
                        # Draw it:
                        cv2.circle(grayscaleImageCopy, (pointX, pointY), 3, color, -1)
                        cv2.imshow("End Points", grayscaleImageCopy)
                        cv2.waitKey(0)
                        # Store the tip and tail points:
                        p0x = orderedPoints[1][0]
                        p0y = orderedPoints[1][1]
                        p1x = orderedPoints[0][0]
                        p1y = orderedPoints[0][1]
                        # Compute the sides of the triangle:
                        adjacentSide = p1x - p0x
                        oppositeSide = p0y - p1y
                        # Compute the angle alpha:
                        alpha = math.degrees(math.atan(oppositeSide / adjacentSide))

                        # Adjust angle to be in [0,360]:
                        if adjacentSide < 0 < oppositeSide:
                            alpha = 180 + alpha
                        else:
                            if adjacentSide < 0 and oppositeSide < 0:
                                alpha = 270 + alpha
                            else:
                                if adjacentSide > 0 > oppositeSide:
                                    alpha = 360 + alpha
                                    # Deep copy for rotation (if needed):
                                    rotatedImg = croppedImg.copy()
                                    # Undo rotation while padding output image:
                                    rotatedImg = rotateBound_v7(rotatedImg, alpha)
                                    cv2.imshow("rotatedImg", rotatedImg)
                                    cv2.waitKey(0)

                else:
                    print("K-Means did not return enough points, skipping...")
            else:
                print("Did not find enough end points on image, skipping...")


def arrow_angle_v4(img_path):
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    # Image preprocessing: binary thresholding
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area to keep the largest one (assuming it's the arrow)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Compute image moments of the largest contour to calculate orientation
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Avoid noise and small components
            M = cv2.moments(cnt)

            # Calculate orientation using moments
            if M["m00"] != 0:
                delta_x = 2 * M["m11"] / M["m00"]
                delta_y = M["m20"] - M["m02"]
                angle = 0.5 * math.atan2(delta_x, delta_y)
                angle_degrees = math.degrees(angle)
                print("Angle in degrees: ", angle_degrees)

    # Assume cnt is the contour and angle_degrees is the angle in degrees
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Taking two points near the ends of the arrow contour
    point1 = tuple(cnt[cnt[:,:,0].argmin()][0])
    point2 = tuple(cnt[cnt[:,:,0].argmax()][0])

    # Calculating distances from centroid to the end points
    dist1 = math.sqrt((cX - point1[0])**2 + (cY - point1[1])**2)
    dist2 = math.sqrt((cX - point2[0])**2 + (cY - point2[1])**2)

    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_img, [cnt], -1, (0, 255, 0), 3)
    cv2.circle(color_img, (cX, cY), 10, (0, 255, 0), -1)

    # Identifying the arrow direction START - BLUE, END - RED
    if dist1 > dist2:
        print(f"The arrow starts at {point2} and ends at {point1}")
        cv2.circle(color_img, point1, 5, (0, 0, 255), -1)
        cv2.circle(color_img, point2, 5, (255, 0, 0), -1)
    else:
        print(f"The arrow starts at {point1} and ends at {point2}")
        cv2.circle(color_img, point1, 5, (255, 0, 0), -1)
        cv2.circle(color_img, point2, 5, (0, 0, 255), -1)

    cv2.imshow('Identified Arrow', color_img)
    border_color = [0, 0, 0]
    border_thickness = 2
    height, width = color_img.shape[:2]
    top_left = (0, 0)
    top_right = (width, 0)
    bottom_left = (0, height)
    bottom_right = (width, height)

    cv2.line(color_img, top_left, top_right, border_color, border_thickness)
    cv2.line(color_img, top_right, bottom_right, border_color, border_thickness)
    cv2.line(color_img, bottom_right, bottom_left, border_color, border_thickness)
    cv2.line(color_img, bottom_left, top_left, border_color, border_thickness)

    cv2.imwrite(f"logs/arrows_plots/{img_path.name}", color_img)

    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def find_head_and_tail_v3(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Apply binary thresholding
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming that the largest contour is the arrow
    cnt = max(contours, key=cv2.contourArea)

    # Calculate the centroid of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = (cX, cY)

    # Calculate the distances of all contour points to the centroid
    distances = np.sqrt((cnt[:, :, 0] - cX) ** 2 + (cnt[:, :, 1] - cY) ** 2)

    # The head is the point nearest to the centroid
    head = tuple(cnt[distances.argmin()][0])

    # The tail is the point farthest from the centroid
    tail = tuple(cnt[distances.argmax()][0])

    return tail, head


def head_tail_detection_v3(img_path):
    plt.figure(figsize=(10, 10))
    image = cv2.imread(str(img_path))
    tail, head = find_head_and_tail_v3(image)
    xx, xy = tail
    yy, yx = head
    image = cv2.circle(image, (xx, xy), radius=0, color=(0, 255, 0), thickness=5)  # head - red
    image = cv2.circle(image, (yy, yx), radius=0, color=(255, 0, 0), thickness=5)  # tail - green

    # Adding text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_color = (0, 0, 255)
    font_thickness = 1
    tail_text_position = (xx + 10, xy)
    head_text_position = (yy + 10, yx)
    cv2.putText(image, 'Tail', tail_text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(image, 'Head', head_text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    border_color = [0, 0, 0]
    border_thickness = 2
    thickness = 3
    height, width = image.shape[:2]
    top_left = (0, 0)
    top_right = (width, 0)
    bottom_left = (0, height)
    bottom_right = (width, height)

    # Rysowanie linii
    cv2.line(image, top_left, top_right, border_color, border_thickness)
    cv2.line(image, top_right, bottom_right, border_color, border_thickness)
    cv2.line(image, bottom_right, bottom_left, border_color, border_thickness)
    cv2.line(image, bottom_left, top_left, border_color, border_thickness)
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(f"logs/arrows_plots/{img_path.name}")
    plt.show()

def find_arrow_PCA_v5(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    # Apply binary thresholding
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming that the largest contour is the arrow
    cnt = max(contours, key=cv2.contourArea)

    # Concatenate all contours
    data = np.vstack([cnt.reshape(-1, 2).astype(np.float32) for cnt in contours])

    mean, eigenvectors = cv2.PCACompute(data, mean=None)

    # The principal eigenvector gives the direction
    principal_eigenvector = eigenvectors[0]
    print(f"Direction: {principal_eigenvector}")

    # Convert to matplotlib color space (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Find centroid of the arrow
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Draw the principal direction as a line from centroid
    start_point = (cX, cY)
    end_point = (int(cX + 100 * principal_eigenvector[0]), int(cY + 100 * principal_eigenvector[1]))
    color = (0, 0, 255)
    border_color = [0, 0, 0]
    border_thickness = 2
    thickness = 3
    height, width = img_rgb.shape[:2]
    cv2.line(img_rgb, start_point, end_point, color, thickness)
    top_left = (0, 0)
    top_right = (width, 0)
    bottom_left = (0, height)
    bottom_right = (width, height)

    cv2.line(img_rgb, top_left, top_right, border_color, border_thickness)
    cv2.line(img_rgb, top_right, bottom_right, border_color, border_thickness)
    cv2.line(img_rgb, bottom_right, bottom_left, border_color, border_thickness)
    cv2.line(img_rgb, bottom_left, top_left, border_color, border_thickness)

    plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(rf"logs\arrows_plots\PCA_TEST\{Path(img_path).name}")
    plt.show()


def filter_contours_v6(image, min_aspect_ratio=2, min_area=50):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_cnt = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        if aspect_ratio > min_aspect_ratio and cv2.contourArea(cnt) > min_area:
            filtered_cnt.append(cnt)

    # Assuming the largest remaining contour is the arrow
    if filtered_cnt:
        cnt = max(filtered_cnt, key=cv2.contourArea)
        cv2.drawContours(image, [cnt], 0, 255, -1)

        return True, image  # Successfully found the arrow contour
    else:
        return False, image  # No suitable arrow contour found


def get_arrow_angle_v6(img_path):
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    # Image preprocessing: binary thresholding
    _, img = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU)
    cv2.imshow('after binary image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    success, filtered_image = filter_contours_v6(img)
    print(success)
    cv2.imshow('after binary image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # get white points
    pnts = cv2.findNonZero(img)
    # min area rect
    rect_center, size, angle = cv2.minAreaRect(pnts)

    # fit line to get angle
    [vx, vy, x, y] = cv2.fitLine(pnts, cv2.DIST_L12, 0, 0.01, 0.01)
    angle = (math.atan2(vy, -vx)) * 180 / math.pi

    M = cv2.moments(img)
    gravity_center = (M["m10"] / M["m00"], M["m01"] / M["m00"])

    angle_vec = (int(gravity_center[0] + 100 * vx), int(gravity_center[1] + 100 * vy))

    # cc_vec = gravity center - rect center
    cc_vec = (gravity_center[0] - rect_center[0], gravity_center[1] - rect_center[1])

    # if dot product is positive add 180 -> angle between [0, 360]
    dot_product = cc_vec[0] * angle_vec[0] + cc_vec[1] * angle_vec[1]
    angle += (dot_product > 0) * 180

    angle += (angle < 0) * 360

    color1 = (255, 0, 0)
    color2 = (0, 255, 0)
    # draw rect center
    cv2.circle(img, (int(rect_center[0]), int(rect_center[1])), 3, color1, -1)
    cv2.circle(img, (int(gravity_center[0]), int(gravity_center[1])), 3, color2, -1)

    print("Angle = ", angle)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    arrow_root = Path(r"logs/arrows")
    for img_path in glob.glob(fr'{arrow_root}/*.jpg'):
        print(img_path)
        if str(img_path).endswith('jpg'):
            # arrowhead_detection_v1(img_path)
            # components_detection_v2(img_path)
            find_arrow_PCA_v5(img_path)
            # arrow_angle_v4(img_path)
            # head_tail_detection_v3(img_path)
            # get_arrow_angle_v6(img_path) # INVALID
            # arrow_v7(img_path) # INVALID

