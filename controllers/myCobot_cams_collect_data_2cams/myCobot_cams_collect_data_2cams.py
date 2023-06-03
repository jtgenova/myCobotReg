"""myCobot_reg controller."""

from controller import Supervisor
import os
import math
import ikpy
from ikpy.chain import Chain
import numpy as np
import torch
from PIL import Image as im
from automate_waypoints import automate
from csv import writer

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

IKPY_MAX_ITERATIONS = 64

# create the Robot instance.
supervisor = Supervisor()

# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())*512

# Create the arm chain from the URDF
base_path = '/home/jtgenova/Documents/GitHub/myCobot_cams/'
os.chdir(base_path)
filename = "myCobot.urdf"
# print(filename)
armChain = Chain.from_urdf_file(filename)
for i in [0, 7]:
    armChain.active_links_mask[i] = False

# Initialize the arm motors and encoders.
motors = []
for link in armChain.links:
    if 'motor' in link.name:
        motor = supervisor.getDevice(link.name)
        motor.setVelocity(1.0)
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timestep)
        motors.append(motor)

# initialize cameras
cam_top = supervisor.getDevice('camera_top')
cam_ee = supervisor.getDevice('camera_side')
cam_top.enable(timestep)
cam_ee.enable(timestep)

# getDef of objects
target = supervisor.getFromDef('waypoint')
feature = supervisor.getFromDef('feature')
end_effector = supervisor.getFromDef('ee_tool')
robot = supervisor.getFromDef('robot')
arm = supervisor.getSelf()

crack_img = supervisor.getFromDef('img_appr')
trans = crack_img.getField('transparency')
img_texture = supervisor.getFromDef('img_texture')
img_url = img_texture.getField('url')
img_url.setMFString(0, " ")

# get waypoint center position
targetPosition = target.getPosition()
waypoint = target.getField('translation')
waypoint_distance = []

sample_episodes = 1

total_steps = 0

supervisor.step(timestep)

# start collecting data
for ep in range(sample_episodes):
    # Reset the simulation
    supervisor.simulationResetPhysics()
    supervisor.simulationReset()
    trans.setSFFloat(1)
    supervisor.step(timestep)
    # create waypoints from mask
    index = np.random.randint(0, 20)
    mask_path = f"{base_path}crack_dataset/mask/"
    mask_images = os.listdir(mask_path)
    mask_path = f"{mask_path}{mask_images[index]}"
    orientation = mask_images[index].split("_")[0]
    a = automate(mask_path, orientation)
    webots_x, webots_y = a.generate_waypoints()
    idx = 0
    waypoint.setSFVec3f([webots_x[idx], webots_y[idx], 0])

    # output image on webots 
    img_path = f"{base_path}crack_dataset/edit/"
    images = os.listdir(img_path)
    img_url.setMFString(0, f"{img_path}{images[index]}")
    
    # randomize position of crack texture
    feature_position = feature.getField('translation')
    feature_x = round(np.random.uniform(-0.03, 0.03), 3)
    feature_y = round(np.random.uniform(0.02, 0.0375), 3)
    feature_rand_position = [feature_x, feature_y, 0.0025]
    feature_position.setSFVec3f(feature_rand_position)
    
    trans.setSFFloat(0)

    max_steps = 50
    
    waypoint_distance = []
    supervisor.step(timestep)

    for steps in range(max_steps):
        total_steps += 1
        
        if idx == len(webots_x):
            print(f"Episode: {ep}, Steps: {steps}")
            print(f"Total Steps = {total_steps}")
            break
        
        os.chdir(base_path)
        cam_top.getImage()
        cam_ee.getImage()
        cam_top.saveImage(f'pics/top/top_{steps}.jpeg', 80)
        cam_ee.saveImage(f'pics/side/ee_{steps}.jpeg', 80)
        
        img_top = torch.from_numpy(np.frombuffer(cam_top.getImage(), dtype=np.uint8).reshape(1, 4, 240, 320)).float()
        img_side = torch.from_numpy(np.frombuffer(cam_ee.getImage(), dtype=np.uint8).reshape(1, 4, 320, 240)).float()

        # Flatten the image tensor into a 1D array
        img_top = img_top.flatten().numpy()
        img_side = img_side.flatten().numpy()

        current_joint_angles = np.zeros(6)
        for i in range(len(motors)):
            current_joint_angles[i] = motors[i].getPositionSensor().getValue()
        
        supervisor.step(timestep)
        
        # Get the absolute postion of the target and the arm base.
        targetPosition = target.getPosition()
        armPosition = arm.getPosition()
        eePosition = end_effector.getPosition()

        # Compute the position of the target relatively to the arm.
        # x and y axis are inverted because the arm is not aligned with the Webots global axes.
        x = targetPosition[0] - armPosition[0]
        y = targetPosition[1] - armPosition[1]
        z = targetPosition[2] - armPosition[2]
        
        distance_x = targetPosition[0] - eePosition[0]
        distance_y = targetPosition[1] - eePosition[1]
        distance_z = targetPosition[2] - eePosition[2]
        
        total_distance = math.sqrt(distance_x**2 + distance_y**2 + distance_z**2)    
        # Call "ikpy" to compute the inverse kinematics of the arm.
        initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
        ikResults = armChain.inverse_kinematics([x, y, z], [0, 0, -1], orientation_mode="Z", max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)

        # Recalculate the inverse kinematics of the arm if necessary.
        position = armChain.forward_kinematics(ikResults)
        squared_distance = (position[0, 3] - x)**2 + (position[1, 3] - y)**2 + (position[2, 3] - z)**2
        if math.sqrt(squared_distance) > 5e-3:
            ikResults = armChain.inverse_kinematics([x, y, z], [0, 0, -1], orientation_mode="Z")
        
        # Actuate the arm motors with the IK results.
        if total_distance > 5e-3:
            for i in range(len(motors)):
                motors[i].setPosition(ikResults[i + 1])
            
        goal_joint_angles = ikResults[1:7]
        
        if total_distance < 5e-3:
            waypoint.setSFVec3f([webots_x[idx], webots_y[idx], 0])
            idx += 1
            
        # write to csv
        os.chdir(base_path)
        list_side = [img_top, img_side, current_joint_angles, goal_joint_angles]
        
        with open('2_cam_OG.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(list_side)
            f_object.close()
        
            

