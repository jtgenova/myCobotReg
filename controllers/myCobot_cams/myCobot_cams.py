"""myCobot_reg controller."""

from controller import Supervisor
import os
import math
import ikpy
from ikpy.chain import Chain
import numpy as np
import torch
from PIL import Image as im
from reg_model import DictModel
from automate_waypoints import automate
from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

IKPY_MAX_ITERATIONS = 64

# create the Robot instance.
supervisor = Supervisor()

# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())*128

# Create the arm chain from the URDF
base_path = '/home/jtgenova/Documents/GitHub/myCobot/'
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

sample_episodes = 100
top_images = []
side_images = []
current_joint_angles = []
goal_joint_angles = []

supervisor.step(timestep)

# start collecting data
for ep in range(sample_episodes):
    # Reset the simulation
    supervisor.simulationResetPhysics()
    supervisor.simulationReset()
    trans.setSFFloat(1)
    supervisor.step(timestep)
    # create waypoints from mask
    index = np.random.randint(0, 10)
    mask_path = f"{base_path}dataset/train/mask/"
    mask_images = os.listdir(mask_path)
    mask_path = f"{mask_path}{mask_images[index]}"
    orientation = mask_images[index].split("_")[0]
    a = automate(mask_path, orientation)
    webots_x, webots_y = a.generate_waypoints()
    idx = 0
    waypoint.setSFVec3f([webots_x[idx], webots_y[idx], 0])

    # output image on webots 
    img_path = f"{base_path}dataset/train/edit/"
    images = os.listdir(img_path)
    img_url.setMFString(0, f"{img_path}{images[index]}")
    
    # randomize position of crack texture
    feature_position = feature.getField('translation')
    feature_x = round(np.random.uniform(-0.02, 0.02), 3)
    feature_y = round(np.random.uniform(0.02, 0.0375), 3)
    feature_rand_position = [feature_x, feature_y, 0.0025]
    feature_position.setSFVec3f(feature_rand_position)
    
    trans.setSFFloat(0)

    max_steps = 50
    
    waypoint_distance = []
    supervisor.step(timestep)

    for steps in range(max_steps):
        print(f"episode: {ep}, steps: {steps}")
        if idx == len(webots_x):
            break
        img_top = torch.from_numpy(np.frombuffer(cam_top.getImage(), dtype=np.uint8).reshape(1, 4, 240, 320)).float()
        img_side = torch.from_numpy(np.frombuffer(cam_ee.getImage(), dtype=np.uint8).reshape(1, 4, 320, 240)).float()
        top_images.append(img_top)
        side_images.append(img_side)
        
        joint_angles = np.zeros(6)
        for i in range(len(motors)):
            joint_angles[i] = motors[i].getPositionSensor().getValue()
        current_joint_angles.append([joint_angles])
        
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
        if math.sqrt(squared_distance) > 1e-3:
            ikResults = armChain.inverse_kinematics([x, y, z], [0, 0, -1], orientation_mode="Z")
        
        # Actuate the arm motors with the IK results.
        if total_distance > 1e-3:
            for i in range(len(motors)):
                motors[i].setPosition(ikResults[i + 1])
        
        joint_angles = np.zeros(6)
        for i in range(len(motors)):
            joint_angles[i] = motors[i].getPositionSensor().getValue()
        goal_joint_angles.append([joint_angles])
        
        if total_distance < 5e-3:
            waypoint.setSFVec3f([webots_x[idx], webots_y[idx], 0])
            idx += 1
            # waypoint_distance.append(total_distance)

# split training and validation data
split_index = int(len(current_joint_angles) * 0.8)

train_top_images = top_images[:split_index]
val_top_images = top_images[split_index:]

train_side_images = side_images[:split_index]
val_side_images = side_images[split_index:]

train_current_joint_angles = current_joint_angles[:split_index]
val_current_joint_angles = current_joint_angles[split_index:]

train_goal_joint_angles = goal_joint_angles[:split_index]
val_goal_joint_angles = goal_joint_angles[split_index:]

print(f"train size = {len(train_current_joint_angles)}")
print(f"val size = {len(val_current_joint_angles)}")
##################################################################################################
# load model and send to device
exp_name = 'exp_2_-epochs_10_-eps_100'
writer = SummaryWriter(f"reg/reg_log/{exp_name}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DictModel(6).to(DEVICE)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
best_loss = float('inf')  # Initialize the best loss

for epoch in range(num_epochs):
    model.train()
    for i in range(len(train_current_joint_angles)):
        optimizer.zero_grad()
        x = {"Top Image": train_top_images[i], "Side Image": train_side_images[i], "Current joint Angles": train_current_joint_angles[i]}
        output = model(x)
        target_joint_angles = torch.Tensor(train_goal_joint_angles[i]).to(DEVICE)
        loss_train = loss(output, target_joint_angles)
        loss_train.backward()
        optimizer.step()
        print("Loss: ", loss_train)
        print("Output: ", output) 

    # Validation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for k in range(len(val_current_joint_angles)):
            x = {"Top Image": val_top_images[k], "Side Image": val_side_images[k], "Current joint Angles": val_current_joint_angles[k]}
            output = model(x)
            target_joint_angles = torch.Tensor(val_goal_joint_angles[k]).to(DEVICE)
            loss_val = loss(output, target_joint_angles)
            total_loss += loss_val
        average_loss = total_loss / len(val_current_joint_angles)

    # Save the model if the current loss is the best so far
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), f"reg/reg_model/{exp_name}.pth")
    
    writer.add_scalar("Loss/train", loss_train, epoch)
    writer.add_scalar("Loss/validation", loss_val, epoch)

writer.flush()
writer.close()



