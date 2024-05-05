import sys
import numpy as np
import matplotlib.pyplot as plt
import struct
from mpl_toolkits.mplot3d import Axes3D
import time
import math

plt.rcParams['figure.facecolor'] = 'white'

## read and parse .splat file 
def read_file(filename):
    
    print('Parsing ' + filename + ' ...')
    points = []

    file = open(filename, 'rb')
    data = file.read()
    num_of_points = int(len(data) / 32) ## each splat is 32 bytes

    minY = 100
    maxY = -100
    minX = 100
    maxX = -100
    minZ = 100
    maxZ = -100

    data = open(filename, 'rb')
    for i in range(num_of_points):
        point = data.read(32)

        ## parse line
        splat = parse_point(point)
        points.append(splat)

        ## find min + max of x, y, z of point cloud
        position = splat['position']
        if (position[0] < minX):
            minX = position[0] 
        if (position[0] > maxX):
            maxX = position[0]
        if (position[1] < minY):
            minY = position[1] 
        if (position[1] > maxY):
            maxY = position[1]
        if (position[2] < minZ):
            minZ = position[2] 
        if (position[2] > maxZ):
            maxZ = position[2]

    # print(minX, maxX, minY, maxY, minZ, maxZ)
    return points, minX, maxX, minY, maxY, minZ, maxZ


def parse_point(point):

    ## properites: position (3 × float32), scale (3 × float32), color (RGBA with straight alpha, 4 × uint8), rotation (4 × uint8) representing the components of a quaternion, each component c is decoded as (c−128)/128)
    position = struct.unpack('3f', point[0:12])
    scale = struct.unpack('3f', point[12:24])
    color = struct.unpack('4B', point[24:28])
    rotation = struct.unpack('4b', point[28:32])
    rotation_decoded = [(c - 128) / 128 for c in rotation]

    return {
        'position': position,
        'scale': scale,
        'color': color,
        'rotation': rotation_decoded
    }

## visualize coordinates of point cloud - not in scope of homework, just for visualization
def plot3D(splats, colors, camera, target):
    print('Visualizing point cloud ...')
    x_coords = splats[:, 0] 
    y_coords = splats[:, 1]
    z_coords = splats[:, 2]

    colors_r = colors[:, 0]
    colors_g = colors[:, 1]
    colors_b = colors[:, 2]
    colors_r = colors_r / 255
    colors_g = colors_g / 255
    colors_b = colors_b / 255
    colors = [(colors_r[i], colors_g[i], colors_b[i]) for i in range(len(splats))]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(x_coords, y_coords, z_coords, c=colors, marker='o')
    ax1.scatter(camera[0], camera[1], camera[2], c='red', marker='x') # optional - plot coordinates of camera in world space
    ax1.scatter(target[0], target[1], target[2], c='black', marker='o') # optional - plot coordinates of target in world space

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    plt.show()

## optional affine transformations of the model
def affine_transformation():
    
    pass

## view transformation matrix - from world space to camera space
def view_transformation(camera_position, target_point):
    print('Preparing view transformation matrix ...')
    ## calculate camera's forward, up and right direction
    ## transformation matrix is inverse of translation, rotation of camera
    up = np.array([0, 1, 0]) # assumed up direction
    forward = (target_point - camera_position) / np.linalg.norm(target_point - camera_position)
    right = np.cross(forward, up) / np.linalg.norm(np.cross(forward, up))  # cross product of forward and up direction
    up = np.cross(right, forward) / np.linalg.norm(np.cross(right, forward))

    ## view transformation matrix 
    view_transformation_matrix = np.array([
        [right[0], up[0], -forward[0], 0],
        [right[1], up[1], -forward[1], 0],
        [right[2], up[2], -forward[2], 0],
        [-np.dot(right, camera_position), -np.dot(up, camera_position), np.dot(forward, camera_position), 1]
    ])

    # view_matrix = np.array([
    #     [right[0], right[1], right[2], -np.dot(right, camera_position)],
    #     [up[0], up[1], up[2], -np.dot(up, camera_position)],
    #     [forward[0], forward[1], forward[2], -np.dot(forward, camera_position)],
    #     [0, 0, 0, 1]
    # ])
    
    return view_transformation_matrix

## perspective transformation matrix - from camera space to clip space (NDC)
def perspective_transformation(fov, aspect, near, far):
    print('Preparing perspective transformation matrix ...')   
    
    perspective_transformation_matrix = np.array([
        [1 / aspect*(np.tan(np.deg2rad(fov) / 2)), 0, 0, 0],
        [0, 1 / np.tan(np.deg2rad(fov) / 2), 0, 0],
        [0, 0, -((far + near) / (far - near)), -((2 * far * near) / (far - near))],
        [0, 0, -1, 0]
    ])

    return perspective_transformation_matrix

def to_homogenous_coords(splats):
    homogenous_coordinates = np.hstack((splats, np.ones((splats.shape[0], 1))))
    return homogenous_coordinates

def back_from_homogenous_coords(homogenous_splats):
    homogeneous_component = homogenous_splats[:, -1]
    heterogeneous_coordinates = homogenous_splats[:, :-1] / homogeneous_component[:, None]
    return heterogeneous_coordinates

def splats_np(splats):
    splats_position = [splat['position'] for splat in splats]
    splats_position = np.array(splats_position)
    return splats_position

def splats_color_np(splats):
    splats_color = [splat['color'] for splat in splats]
    splats_color = np.array(splats_color)
    return splats_color

def gaussian_function(x, c, sigma):
    distance = x - c
    exponent = -0.5 * np.dot(distance.T, np.dot(sigma, distance))
    return np.exp(exponent)




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Please use a single argument when running the program!')
        print('Run the program with "python gaussian-splatting.py <filename.splat>"')
        sys.exit(1)
    
    """
    (2.0) input
    """
    run_start_time = time.time()

    filename = sys.argv[1]
    splats, minX, maxX, minY, maxY, minZ, maxZ = read_file(filename)

    splats_position = splats_np(splats) # numpy array with just coordinates of splats
    colors = splats_color_np(splats) # numpy array with just colors of splats
    
    

    """
    (3.1) basic transformation and point rendering
    """
    
    ## define camera parameters - position, direction, pov, near and far field, aspect ratio, screen_size,
    
    
    camera_position = np.array([0, -5, -4])
    target_point = np.array([0, 0, 0]) # point in world space where camera is facing

    # camera_position = np.array([5, 100, -70])
    # target_point = np.array([0, 0, 0]) # point in world space where camera is facing

    # plot3D(splats_position, colors, camera_position, target_point) # plot splats in world space

    camera_fov = 90
    near_clip = 0.1
    far_clip = 100
    screen_size = (512, 512)
    aspect_ratio = 1 / 1

    ## view transformation matrix
    view_transformation = view_transformation(camera_position, target_point)
    ## perspective transformation matrix
    perspective_transformation = perspective_transformation(camera_fov, aspect_ratio, near_clip, far_clip)

    ## view transformation
    print('Computing transformations ...')
    splats_position_to_homogenous = to_homogenous_coords(splats_position) # numpy array with homogenous coordinates of splats in world space
    splats_camera_space = np.matmul(splats_position_to_homogenous, view_transformation) 
    splats_camera_space = back_from_homogenous_coords(splats_camera_space) # numpy array of splats in camera space back from homogenous coordinates
    # print(splats_camera_space)
    # plot3D(splats_camera_space, colors, [0, 0, 0]) # splats plotted in 3D in ccs

    ## perspective transformation
    splats_camera_space_to_homogenous = to_homogenous_coords(splats_camera_space) # numpy array with homogenous coordinates of splats in camera space
    splats_clip_space = np.matmul(splats_camera_space_to_homogenous, perspective_transformation)
    splats_clip_space = back_from_homogenous_coords(splats_clip_space) # numpy array of splats in normalized device space back from homogenous coordinates
    
    transformations_end_time = time.time()
    # plot3D(splats_clip_space, colors, [0, 0, 0]) # splats plotted in 3D in clip space - or NDC
    # exit()
        
    ## render splats in screen space
    print('Rendering pixel image for task (3.1) Basic transformations and point rendering ...')
    image = np.zeros((screen_size[0], screen_size [1], 3), dtype=np.uint8)
    image[:, :] = (255, 255, 255)
    i = 0
    for splat in splats_clip_space:
        # print(splat)
        screen_x = (splat[0] + 1) * screen_size[0] / 2
        screen_y = (1 - splat[1]) * screen_size[1] / 2
        color = colors[i]
        
        ## we leave out alpha at this step, only show colors
        r, g, b = color[0], color[1], color[2]
        
        # print(screen_x, screen_y)
        if 0 <= int(screen_x) < screen_size[0] and 0 <= int(screen_y) < screen_size[1]:
            image[int(screen_x), int(screen_y)] = (r,g,b)
        i = i + 1
    plt.figure(facecolor='white', num="(3.1) Basic transformations and point rendering")
    plt.title('Basic pixel rendering')
    plt.imshow(image)
    plt.show(block=False)
    
    task_1_end_time = time.time()
    elapsed_time_task_1 = task_1_end_time - run_start_time
    print("Time spent on (3.1) Basic transformations and point rendering: {:.2f} seconds".format(elapsed_time_task_1))

    """
    (3.2) perspective-correct scaling
    """
    ## dynamic imput of scaling parameter s
    scaling_parameter = input("Choose and input a scaling parameter (s) for perspective-correct scaling:\n")

    task_2_start_time = time.time()

    print('Rendering pixel image for task (3.2) Perspective-correct scaling ...')
    image = np.zeros((screen_size[0], screen_size [1], 3), dtype=np.uint8)
    image[:, :] = (255, 255, 255)

    ## z is the depth in view space (before perspective division)
    z_coords = splats_camera_space[:,2]
    # print(z_coords)
   
    i = 0
    unique_values = []
    
    for splat in splats_clip_space:
        # print(splat)
        # x and y from clip to screen space
        screen_x = (splat[0] + 1) * screen_size[0] / 2
        screen_y = (1 - splat[1]) * screen_size[1] / 2
        color = colors[i]
        
        r, g, b = color[0], color[1], color[2]
        z = float(z_coords[i])
        # print(z)
        scaling_parameter = float(scaling_parameter)
        scale = float((scaling_parameter / abs(z)))
        # print(scale)
        a = 2 * scale 

        #pixel coordinates for squares
        start_x = screen_x - a/2
        start_y = screen_y - a/2
        end_x = start_x + a
        end_y = start_y + a

        # print(a)
        # print(start_x, end_x)
        
        unique_values.append(int(end_x)-int(start_x))
        if 0 <= int(screen_x) < screen_size[0] and 0 <= int(screen_y) < screen_size[1]: 
            # print(int(end_x - start_x), int(end_y - start_y))
            image[int(start_x):int(end_x), int(start_y):int(end_y)] = (r,g,b)
        i = i + 1
        
    unique_values = np.unique(unique_values)
    # print(unique_values) # uncomment to see how many unique values of size of splats there are
    plt.figure(facecolor='white',num="(3.2) Perspective-correct scaling")
    plt.title("Perspective-correct scaling with scaling parameter s = " + str(int(scaling_parameter)))
    plt.imshow(image)
    plt.show(block=False)

    task_2_end_time = time.time()
    elapsed_time_task_2 = transformations_end_time - run_start_time + task_2_end_time - task_2_start_time
    print("Time spent on (3.2) Perspective-correct scaling: {:.2f} seconds".format(elapsed_time_task_2))
    
    """
    (3.3) order-correct blending
    """
    task_3_start_time = time.time()

    ## sort splats by z - sort the splats according to the screen-space depth before rendering
    to_be_sorted_splats = splats_clip_space
    ## sort corresponding colors
    to_be_sorted_colors = colors
    
    sorted_indices = np.argsort(to_be_sorted_splats[:, 2])[::-1] # sort by descending depth

    sorted_splats = to_be_sorted_splats[sorted_indices] 
    sorted_colors = to_be_sorted_colors[sorted_indices] # sorted colors corresponding to splats
    # print(sorted_splats)
    # print(sorted_colors)
    
    print('Rendering pixel image for task (3.3) Order-correct blending ...')
    ## initialize the pic with white color and full opacity
    image = np.zeros((screen_size[0], screen_size[1], 4), dtype=np.uint8)
    image[:, :] = (255, 255, 255, 255)

    z_coords = splats_camera_space[:,2]
    
    i = 0
    for splat in sorted_splats:
        # print(splat)
        # x and y from clip to screen space
        screen_x = (splat[0] + 1) * screen_size[0] / 2
        screen_y = (1 - splat[1]) * screen_size[1] / 2
        color = sorted_colors[i]
        
        r, g, b, alpha = color[0], color[1], color[2], color[3]
        alpha = alpha/255
        # print(r, g, b, alpha)
        
        z = float(z_coords[i])
        # print(z)
        
        ## use scaling parameter from before
        scaling_parameter = float(scaling_parameter)
        scale = float((scaling_parameter / abs(z)))
        # print(scale)
        a = 2 * scale 

        ## pixel coordinates for squares
        start_x = screen_x - a/2
        start_y = screen_y - a/2
        end_x = start_x + a
        end_y = start_y + a

        ## calculate blended color
        true_color_splat = np.array([r, g, b])
        true_color_splat = true_color_splat*alpha
        # print(true_color_splat)

        # print(a)
        # print(start_x, end_x)
        # exit()
        if 0 <= int(screen_x) < screen_size[0] and 0 <= int(screen_y) < screen_size[1]:
            # print(int(end_x - start_x), int(end_y - start_y))
            ## use the blending formula and save to corresponding pixels
            image[int(start_x):int(end_x), int(start_y):int(end_y), :3] = (1 - alpha)*image[int(screen_x), int(screen_y), :3] + true_color_splat
        i = i + 1

    task_3_end_time = time.time()
    elapsed_time_task_3 = transformations_end_time - run_start_time + task_3_end_time - task_3_start_time
    

    plt.figure(facecolor='white', num="(3.3) Order-correct blending")
    plt.title("Order-correct blending with scaling parameter s = " + str(int(scaling_parameter)))
    print("Time spent on (3.3) Order-correct blending: {:.2f} seconds".format(elapsed_time_task_3))
    plt.imshow(image[:,:,:3])
    plt.show(block=False)

    """
    (3.4) gaussian falloff
    """

    task_4_start_time = time.time()
    # use sorted splats and sorted colors from previous (3.3) task
    # print(sorted_splats)
    # print(sorted_colors)

    print('Rendering pixel image for task (3.4) Gaussian falloff ...')
    ## initialize the pic with white color and full opacity
    image = np.zeros((screen_size[0], screen_size[1], 4), dtype=np.uint8)
    image[:, :] = (255, 255, 255, 255)

    z_coords = splats_camera_space[:,2]

    i = 0
    for splat in sorted_splats:
        # print(splat)
        # x and y from clip to screen space
        screen_x = (splat[0] + 1) * screen_size[0] / 2
        screen_y = (1 - splat[1]) * screen_size[1] / 2
        color = sorted_colors[i]
        
        r, g, b, alpha = color[0], color[1], color[2], color[3]
        
        # print(r, g, b, alpha)
        color_splat = np.array([r, g, b])
        
        z = float(z_coords[i])
        # print(z)
            
        # sigma = s/z
        scaling_parameter = float(scaling_parameter)
        sigma = float((scaling_parameter / abs(z)))
        a = 2*sigma

        # covariance matrix sigma
        '''
        Σ = [ s/z  0  0
               0  s/z 0
               0   0  s/z ]
        '''
        covariance_matrix_sigma = np.diag([sigma, sigma, sigma])
        # print(covariance_matrix_sigma)     

        

        start_x = int(screen_x) - int(a/2)
        start_y = int(screen_y) - int(a/2)
        end_x = start_x + a
        end_y = start_y + a

        true_color_splat = np.array([r, g, b])
        true_color_splat_alpha = true_color_splat*alpha/255

        # calculate g(x) for each pixel of image and current splat ()
        x_pixels = np.arange(int(start_x), int(end_x) + 1) 
        y_pixels = np.arange(int(start_y), int(end_y) + 1)
        # print(x_pixels)
        # print(y_pixels)

        # varying alpha for pixels on the virtual square
        varying_alpha = []
        
        for x in (x_pixels):
            for y in (y_pixels):
                pixel = np.array([int(x), int(y), 0])
                center = np.array([int(screen_x), int(screen_y), 0])
                # calculate g(x)
                gauss = gaussian_function(pixel, center, covariance_matrix_sigma)
                varying_alpha_pixel = alpha*gauss
                varying_alpha.append(varying_alpha_pixel)
                #gauss_color_splat = true_color_splat*gauss
       
        if 0 <= int(screen_x) < screen_size[0] and 0 <= int(screen_y) < screen_size[1]:
            # print(int(end_x - start_x), int(end_y - start_y))
            ## use the blending formula and save to corresponding pixels
            image[int(start_x):int(end_x), int(start_y):int(end_y), :3] = (1 - varying_alpha[:])*image[int(start_x), int(start_y), :3] + color_splat*varying_alpha[:]

        i = i + 1

    task_4_end_time = time.time()
    elapsed_time_task_4 = transformations_end_time - run_start_time + task_4_end_time - task_4_start_time
    

    plt.figure(facecolor='white', num="(3.4) Gaussian falloff")
    plt.title("Gaussian fallof with scaling parameter s = " + str(int(scaling_parameter)))
    print("Time spent on (3.4) Gaussian falloff: {:.2f} seconds".format(elapsed_time_task_4))
    plt.imshow(image[:,:,:3])
    plt.show()