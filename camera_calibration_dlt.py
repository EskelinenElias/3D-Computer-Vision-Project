import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import h5py
from checkerboard_detection import _find_checkerboard_corners
from PIL import Image
from scipy.linalg import rq
from pathlib import Path



def select_points(img, scale_factor=0.5):
    """Function to select points from an image.
    
    First select a point from the left, then from the right. If you selected all the points you wanted, press Esc or close the window.
    """
    points1 = []
    count = 0  # Use a list to make count mutable within the nested function

    def click_event(event, x, y, flags, param):
        nonlocal count
        if event == cv.EVENT_LBUTTONDOWN:

            original_x = int(x)
            original_y = int(y)
            points1.append((original_x, original_y))

            cv.circle(param, (x, y), 3, (0, 255, 0), -1)
            cv.imshow("Image", param)
            count += 1

    # Resize image for display
    height, width = img.shape[0], img.shape[1]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    resized_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)

    cv.imshow("Image", resized_img)
    cv.setMouseCallback("Image", click_event, resized_img)

    # Stop when esc pressed
    while True:
        key = cv.waitKey(20) & 0xFF
        if key == 27:  # ESC key to break
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:  # Check if window is closed
            break
        if count == 2:
            break
    cv.destroyAllWindows()
    return points1



def pixel_shift(cube_corners):
    bottom_corner = cube_corners[1]
    top_corner    = cube_corners[0]

    # Shift per cm of Z height
    cube_height_cm = 5.0
    shift_per_cm = (top_corner - bottom_corner) / cube_height_cm
    return shift_per_cm
    

def checkerboard_check(calibration_images,PATTERN_SIZE):
    for i, image in enumerate(calibration_images):
        gray = np.array(image.convert("L"), dtype=np.uint8)
        try:
            checker_2d, _ = _find_checkerboard_corners(gray, PATTERN_SIZE)
            print(f"Image {i}: {len(checker_2d)} corners detected")
        except RuntimeError:
            print(f"Image {i}: no corners detected")
            continue
    return(checker_2d)



def construct_3d_coordinates(N):
    # Constructing the 3d points to map a coordination system from the first corner.
    checker_3d = []

    for k in range(N):
        i = np.floor(k/8)
        j = k % 8
        if k % 2 == 0:
            checker_3d.append([j*-4, i*4, 4])
        else:
            checker_3d.append([j*-4, i*4, 0])

    checker_3d = np.array(checker_3d)
    return(checker_3d)


def fabricate_height(N, checker_2d, checker_z):
    # modifying the checker_2d values to reflect the fabricated height changes
    for i in range(N):
        if N % 2 == 0:
            idx = i
            checker_2d[idx,1] -= checker_z
    return(checker_2d)


# Calibrate to find the camera matrix using 2d and 3d points.
def calibrate(points2d, points3d):
    A = np.zeros([2*len(points2d),12])

    # Computing the matrix A
    for i in range(len(points2d)):
        x,y = points2d[i]
        X,Y,Z = points3d[i]
        A[2*i,:]=[X,Y,Z,1,0,0,0,0,-x*X,-x*Y,-x*Z,-x]
        A[2*i+1,:]=[0,0,0,0,X,Y,Z,1,-y*X,-y*Y,-y*Z,-y]

    # Then to minimize 'm' in Am=0
    U,D,V = np.linalg.svd(A)
    m = V[-1,:]
    m = m.reshape(3, 4)
    #print(m.shape)
    return m


def apply_transformation(points3d, m):
    XYZ1 = np.concatenate([points3d, np.ones([points3d.shape[0],1])],axis=1)
    u,v,w = m @ XYZ1.T
    x = u/w
    y = v/w
    points = np.array([x,y]).T
    return points,x,y


def calibrate_norm(points2d, points3d):
    x_tilde = 1/len(points2d) * np.sum(points2d[:,0])
    y_tilde = 1/len(points2d) * np.sum(points2d[:,1])
    d_tilde = 1/len(points2d) * np.sum(np.sqrt((points2d[:,0] - x_tilde)**2 \
                                        + (points2d[:,1] - y_tilde)**2))
    T_2d = [[np.sqrt(2)/d_tilde, 0, -np.sqrt(2) * x_tilde / d_tilde],
            [0, np.sqrt(2)/d_tilde, -np.sqrt(2) * y_tilde / d_tilde],
            [0, 0, 1]
            ]

    x_tilde = 1/len(points3d) * np.sum(points3d[:,0])
    y_tilde = 1/len(points3d) * np.sum(points3d[:,1])
    z_tilde = 1/len(points3d) * np.sum(points3d[:,2])
    d_tilde = 1/len(points3d) * np.sum(np.sqrt((points3d[:,0] - x_tilde)**2 \
                                        + (points3d[:,1] - y_tilde)**2 \
                                        + (points3d[:,2] - z_tilde)**2))


    T_3d = [[np.sqrt(3)/d_tilde, 0, 0, -np.sqrt(3) * x_tilde / d_tilde],
            [0, np.sqrt(3)/d_tilde, 0, -np.sqrt(3) * y_tilde / d_tilde],
            [0, 0, np.sqrt(3)/d_tilde, -np.sqrt(3) * z_tilde / d_tilde],
            [0, 0, 0, 1]
            ]


    # Changed the coordinates to homogenous to be able to multiply with T and U
    points2d_homog = np.concatenate([points2d,np.ones([len(points2d),1])],axis=1)
    points3d_homog = np.concatenate([points3d,np.ones([len(points3d),1])],axis=1)

    # Transposed them to match the dims
    normalized_2d_pts = T_2d @ points2d_homog.T
    normalized_3d_pts = T_3d @ points3d_homog.T

    #print(np.shape(normalized_2d_pts))
    # dehomogenizing the points
    normalized_2d_pts = normalized_2d_pts.T[:,0:2]
    normalized_3d_pts = normalized_3d_pts.T[:,0:3]

    # Calibrating with the same function
    M = calibrate(normalized_2d_pts, normalized_3d_pts)
    
    # Denormalize
    denormalized_M = np.linalg.inv(T_2d) @ M @ T_3d
    return denormalized_M


def decompose_projection(M):
    K,R = rq(M[:, 0:3])

    X = np.linalg.det(M[:,[1,2,3]])
    Y = -np.linalg.det(M[:,[0,2,3]])
    Z = np.linalg.det(M[:,[0,1,3]])
    W = -np.linalg.det(M[:,[0,1,2]])

    C = np.array([X/W,Y/W,Z/W])

    # rq decomposition can throw a weird result, this make sure that the result is valid for our purposes
    R = R * np.sign(K[-1,-1])
    K = K * np.sign(K[-1,-1])

    return K, R, C




def calibrate_DLT(extrinsic_img_input):
    PATTERN_SIZE = (8, 6)

    # ── Accept either a file path OR a PIL Image ───────────────────────────
    if isinstance(extrinsic_img_input, (str, Path)):
        extrinsic_img_cv  = cv.imread(str(extrinsic_img_input), cv.IMREAD_COLOR_RGB)
        extrinsic_img_pil = Image.open(extrinsic_img_input)
    else:
        # Already a PIL Image
        extrinsic_img_pil = extrinsic_img_input
        extrinsic_img_cv  = cv.cvtColor(np.array(extrinsic_img_pil), cv.COLOR_RGB2BGR)
        extrinsic_img_cv  = cv.cvtColor(extrinsic_img_cv, cv.COLOR_BGR2RGB)

    # SELECT BOTTOM -> TOP OF A CORNER
    cube_corners = select_points(extrinsic_img_cv)
    cube_corners = np.array(cube_corners) * 2

    shift_per_cm = pixel_shift(cube_corners)
    checker_z = 4 * shift_per_cm[1]

    # Finding the checkerboard spots
    checker_2d = checkerboard_check([extrinsic_img_pil], PATTERN_SIZE)
    N = len(checker_2d)

    # Fabricating the height to the 2d coordinates, making the squares "cubes"
    checker_2d = fabricate_height(N, checker_2d, checker_z)

    # Constructing the 3d coordinates to match the fabricated 2d, and use first spot as origin
    checker_3d = construct_3d_coordinates(N)

    # Then finally calibrating and returning the needed values
    M = calibrate_norm(checker_2d, checker_3d)
    K, R, C = decompose_projection(M)
    print("done")
    return M, K, R, C