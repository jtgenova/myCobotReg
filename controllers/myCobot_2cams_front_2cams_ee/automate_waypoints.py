from skimage.morphology import skeletonize
from statistics import mean
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import os

class automate:
    def __init__(self, image_path, orientation):
        self.image_path = image_path
        self.orientation = orientation
        self.size = 750
        self.waypoint_x =[]
        self.waypoint_y = []
        self.webots_x = []
        self.webots_y = []

    def generate_waypoints(self):
        image = Image.open(self.image_path)
        image = image.resize((self.size, self.size))
        self.np_img = np.asarray(image)
        # perform skeletonization
        skeleton = skeletonize(self.np_img)

        if self.orientation == 'vertical':
            rows, cols = np.where(skeleton)
            coords = list(zip(rows, cols))
            coords.sort(key=lambda x: x[0])
            y, x = list(zip(*coords))
        
        if self.orientation == 'horizontal':
            cols, rows = np.where(skeleton)
            coords = list(zip(rows, cols))
            coords.sort(key=lambda x: x[0])
            x, y = list(zip(*coords))
        
        waypoint_idx = math.trunc(len(x) / 10)

        self.waypoint_x =[]
        self.waypoint_y = []
        webots_x = []
        webots_y = []
        for i in range(0, len(x), waypoint_idx):
            self.waypoint_x.append(mean(x[i:i+waypoint_idx]))
            self.waypoint_y.append(mean(y[i:i+waypoint_idx]))
            webots_x.append(round(mean(x[i:i+waypoint_idx])/1e4 - 0.0375, 4)) 
            webots_y.append(round(mean(y[i:i+waypoint_idx])/1e4 - 0.0375,  4))

        return webots_x, webots_y
    
    def show_plot(self):
        webots_x, webots_y = self.generate_waypoints()
        im = plt.imshow(self.np_img)
        # implot = plt.imshow(im)
        plt.plot(self.waypoint_x, self.waypoint_y, 'o')
        plt.xlim([0, 750])
        plt.ylim([0, 750])
        plt.show()

        plt.plot(webots_x, webots_y, 'o')
        plt.xlim([-0.0375, 0.0375])
        plt.ylim([-0.0375, 0.0375])
        plt.show()


def create_dict(path):
    img_list = os.listdir(path)
    my_dict = {"ID": [], "Image Path": [], "X": [], "Y": []}
    for i in range(len(os.listdir(path))):
        image_path = f'{path}/{img_list[i]}'
        name, ext = img_list[i].split('.')
        orientation, ext = name.split('_')
        id = i+ 1

        my_dict["ID"].append(id)
        my_dict["Image Path"].append(image_path)
        auto = automate(image_path, orientation)
        print(image_path)
        webots_x, webots_y = auto.generate_waypoints()
        auto.show_plot()
        my_dict["X"].append(webots_x)
        my_dict["Y"].append(webots_y)

        # print(f"ID: {my_dict['ID'][i]}")
        # print(f"Image Path: {my_dict['Image Path'][i]}")
        # print(f"X: {my_dict['X'][i]}")
        # print(f"Y: {my_dict['Y'][i]}\n")

def edit_image(path):
    size = 512
    img_list = os.listdir(path)
    for i in range(len(img_list)):
        image_path = f'{path}/{img_list[i]}'
        name = img_list[i]
        #read the image
        im = Image.open(image_path)
        orientation = img_list[i].split("_")[0]
        # if orientation == "vertical":
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        if orientation == "horizontal":
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = im.resize((size, size))
        
        im.save(f'C:/Users/Administrator/Documents/GitHub/myCobot/crack_images/resize/{name}')


if __name__=="__main__":
        path = "C:/Users/Administrator/Documents/GitHub/myCobot/crack_images/images/"
        edit_image(path)
        # images = os.listdir(path)
        # orientation = images[0].split("_")[0]
        # a = automate(f"{path}{images[0]}", orientation)
        # a.show_plot()


   
