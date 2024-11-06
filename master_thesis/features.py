import csv
import numpy as np
import yaml
import os
from scipy.interpolate import PchipInterpolator, interp1d
import matplotlib.pyplot as plt
from math import ceil
import colorsys
import random


### --------- READING FROM FILES

# Step 1: i have to read from the files to get the data
# should read from a yaml file probably....
def read_csv(filename):
    """In this method we read the filename containing joint limits
    and put them in a list."""

    limit_dict = {}

    with open(filename, "r") as file:
        reader = csv.reader(file)

        for row in reader:
            limit_dict[row[0]] = [float(row[1]), float(row[2])]

    
    return limit_dict


# Step 2: find feature 1
# angle percentage (percentage of max angle) using: http://doc.aldebaran.com/2-8/family/nao_technical/joints_naov6.html

def read_motion(filename):
    """Reads the motion text file and returns the dictionary containing some aspects of the motion,
    like the joint angles and end-effector positions."""

    with open(filename, "r") as file:
        motion = yaml.safe_load(file)

    # returns the whole yaml file in a dictionary
    return motion



### ------------------- MAPPING OF ANGLES

def cubic_interpolation(nao, misty, init_nao, init_misty, pos): 
    """Takes in the limits and the position we want interpolated"""
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    
    # lin = interp1d(nao, misty)
    # interpolated_value1 = lin(pos)
        
    #adding the init position as a point
    nao.append(init_nao)
    nao.sort()
    misty.append(init_misty)
    misty.sort()
    
    
    pchip = PchipInterpolator(nao, misty)
    interpolated_value = pchip(pos)
    

        
    # return interpolated_value.tolist(), interpolated_value1.tolist()
    return interpolated_value.tolist()

def map_angles_misty(mappings, motion_dict, scene, misty_limits, nao_limits):
    """This function will decide the movement of mistys hands based on values from nao head
    for a certain movement and then this will be appended to an output file. 
    This file does it only for one scene and appends it to its own yaml file."""
    
    # this is joint angle - joint angle
    
    # here you go through only the angles in the file
    
    values = {} # creating an empty dictionary where we can append values
    
    
    for joint in mappings: # going through the joints involved in the mapping to find the limits
        nao_joint = mappings[joint]
        min_misty, max_misty = misty_limits[joint][0], misty_limits[joint][1] 
        min_nao, max_nao = nao_limits[nao_joint][0], nao_limits[nao_joint][1]
        

        if joint == "LArm" or joint == "RArm": # da er init basert på armane til misty
            init_misty = np.deg2rad(70) # dette er initen til armene
        else:
            init_misty = 0
      
        angle_list = motion_dict[nao_joint]
        
        init_nao = angle_list[0]
             
        # linear conversion formula: https://www.cuemath.com/linear-equation-formula/ - kommer herfra
        #new_angle = ((angle-min_nao)*(max_misty-min_misty)) / (max_nao-min_nao) + min_misty
            
        # we use cubic interpolation here too cause we want to also use the init pos
        new_angle = cubic_interpolation([min_nao, max_nao], [min_misty, max_misty], init_nao, init_misty, angle_list)
        values[joint] = new_angle

    
        # nr = np.linspace(0, len(angle_list), len(angle_list))
        # plt.title(f"{scene} - {joint}")
        # plt.plot(nr, angle_list, color="#D81B60", label="NAO")
        # plt.plot(nr, new_angle, color="#1E88E5", label="Misty")
        # plt.axhline(y=min_nao, linestyle="--", color="#D81B60", label="NAO limits")
        # plt.axhline(y=max_nao, linestyle="--", color="#D81B60")
        
        # plt.axhline(y=min_misty, linestyle="--", color="#1E88E5", label="Misty limits")
        # plt.axhline(y=max_misty, linestyle="--", color="#1E88E5")
        # plt.legend()
        # plt.xlabel("Time (s)")
        # plt.ylabel("Angle (rad)")
        # plt.show()
           
    return values # this returns a dictionary 


def turning_points_arms(vals):
    
    """vals is a list and i need to call it for each component that has a list: 
    component : [] inside a dictionary."""
    turning_points = [vals[0]]
    indeces = [0]
    
    # kunne egt brukt scipy find peaks her..... ffs... og thresholda dem samtidig.... FFS
    
    # jumps tell us how many steps there are between each extreme point
    
    # going through the whole list and finding turning points
    for i in range(1, len(vals)-1): # hopper over første og siste
        this, previous, next = vals[i], vals[i-1], vals[i+1]
            
        if (this > previous and this > next) or (this < previous and this < next): # da har vi et extrempunkt
            
            turning_points.append(this)
            indeces.append(round(i/2)) # cause we sample each 0.5 sec and we need the indices to be right
            #indeces.append(i) 
    turning_points.append(vals[-1])
    indeces.append(round((len(vals)-1)/2))
    #indeces.append(i)
    
    
    # p = find_peaks(vals)
    # v = find_peaks(-np.array(vals))
    # new = np.concatenate((p[0],v[0]))
    
    # turning_points_indices = np.sort(np.array(new))
    # turning_points_values = np.array(vals)[turning_points_indices]
    
    # nr = np.linspace(0, len(vals), len(vals))
    # #plt.title(f"{scene} - {key}")
    # plt.plot(nr, vals)
    # plt.scatter(turning_points_indices, turning_points_values)
    # plt.show()
   
    return turning_points, indeces


def processing_angles(nao_limits, misty_limits, motion_dict, scene, mappings, threshold):
    """This will find the values for the arms specifically for mist (map_angles_misty) and then will find the turning points 
    so reduce the list of different commands that we send to misty."""
    
    # mapping: nao head pitch = hands misty - direct mapping ut i fra joint range of head and hands
    # misty -> nao
    
    
    misty_vals = map_angles_misty(mappings, motion_dict, scene, misty_limits, nao_limits)
    
    full = {}
    
    for key in misty_vals: # going through the dictionary
        turning_points, indeces = turning_points_arms(misty_vals[key]) # finding turning points list 
        new_vals, new_indices = check_turning_points(turning_points, indeces, threshold)
 
        #smoothed_list = smooooooth(turning_points, win_size=8)
        
        # instead of smoothing the list we just send the both the turning points and the number of steps between 
        # so that we can choose how fast the movement should be/how long it should last
        #full[key] = [turning_points, indeces]
        full[key] = [new_vals, new_indices]
        
        
    if "Pitch" in full.keys() and "Roll" not in full.keys(): # if roll is not present, then you set it to 0
        full["Roll"] = [[0.0], [0]]
        
    return full



### ------------------- MAPPING OF DANCES
    
def smooooooth(values, win_size):
    # implementere moving average
    full_list = []
    
    full_list.append(values[0])
    
    for i in range(len(values)-win_size+1):
        window = values[i:i+win_size]
        window_average = sum(window)/win_size
        full_list.append(window_average)
    
    full_list.append(values[-1])
    #print("Smoothed list", full_list)
    
    return full_list


def sort_position_data(vector_list):
    
    """Go through the motion dictionary and go through the endeffector_positions
    part of the list for a scene."""
    
    # getting the orientation
    
    all = {"x": [], "y": [], "z": []}
    for lst in vector_list: # list of lists that we have to iterate through
        all["x"].append(lst[0])
        all["y"].append(lst[1])
        all["z"].append(lst[2])
    return all


def direction_vectors(tps, arm=False):
    
    mapping = ["Roll", "Pitch", "Yaw"] 
    directions = {angle: [] for angle in mapping}
    directions["Magnitude"] = []

    for i in range(len(tps)-1): # iterating through all turning points
        dir_vec = np.array(tps[i+1]) - np.array(tps[i])
        
        magnitude = np.linalg.norm(dir_vec)
                   
        sign_list = np.sign(dir_vec) # get the sign of the vector: [x,y,z] 
       
        # pitch: have to change sign because if negative it means arm moves down which means head down, which is positive on misty
        # yaw: if positive the arm moves to the left which is positive and that is the positive dir for misty too
        
        directions["Pitch"].append(int(sign_list[2])) # z axis append sign directly
        directions["Yaw"].append(int(sign_list[1])) # y axis append sign directly too
        directions["Roll"].append(int(sign_list[0])) # x axis append sign directly
    
        directions["Magnitude"].append(magnitude)
        
        # for roll we have to check pairs of z and y axis to get diagonal movement:
        # and we check if roll is in the mapping at all, cause if not we want to map the roll angle to the hands instead, not the head
        # so in this method we only give values to mistys head
        
        # men vi kan jo adde roll anyway også choose to ignore it når vi programma misty.
    
    return directions
    
      
def find_sig_dirs(lst):
    
    displacements = np.diff(lst, axis=0)

    tot_x = float(np.sum(np.abs(displacements[:, 0])))
    tot_y = float(np.sum(np.abs(displacements[:, 1])))
    tot_z = float(np.sum(np.abs(displacements[:, 2])))
    
   
    
    full = {"Roll":tot_x, "Yaw":tot_y, "Pitch":tot_z}
    
    return full


def check_ranges(motion_dict_nao, lst): 
    qom_minimas = motion_dict_nao["qom"]["extreme_points"] # getting where the minima happens
    qom_minimas.sort() # adding the first
    
    new = [[x,y,z] for x,y,z in zip(lst["x"], lst["y"], lst["z"])]
    significant = []
    
    
    for i in range(len(qom_minimas)-1):
        # getting the tilsvarende from new
        look_for1, look_for2 = int(qom_minimas[i]/30)*2, int(qom_minimas[i+1]/30)*2
        interval_xyz = new[look_for1:look_for2] 
       
       # den kjører bare en gang???
        if len(interval_xyz) != 0:
            sig_dir_interval = find_sig_dirs(interval_xyz)
            significant.append([sig_dir_interval, [int(look_for1/2),int(look_for2/2)]])
                
    return significant
        
def linear_mapping(misty_limits, x, dir):
    # bare fra 0 til 4/max cause the minus comes later.
    misty = [misty_limits[dir][0], misty_limits[dir][1]]
    original = [-4, 4]
    
    new = misty[0] + ((x-original[0])*(misty[1]-misty[0]))/(original[1]-original[0])
    return new

def xyz(motion_dict_nao, scene, misty_limits):
    
    print("XYZ\n")
    sort = sort_position_data(motion_dict_nao["endeffector_positions"]["LArm"])
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(sort["x"], sort["y"], sort["z"], color="#1E88E5")
    # ax.set_title(f"{scene} - left arm NAO")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    # plt.show()

    
    sig_dirs = check_ranges(motion_dict_nao, sort)
    
    tp, indices = turning_points(sort["x"], sort["y"], sort["z"]) # find turning points
    # KAN VÆR E MÅ SJEKKE TURNING POINTS først?? eller etter ??
    

    directions = direction_vectors(tp)
    print("DIRS", directions)
    # go through the directions and the magnitudes
    # find the highest magnitude in the list and the lowest and map them to mistys limits and then find the angle and give it the 
    # corresponding sign - and this is what you put into misty
    
    full = {"Roll": [], "Pitch": [], "Yaw": []}
        
    magnitudes = smooooooth(directions["Magnitude"],3) # get the whole list
    steps = (np.max(magnitudes) - np.min(magnitudes))/4 # 4 er for position - 1,2,3,4
    ranges = [[np.min(magnitudes) + i*steps, np.min(magnitudes) + (i+1)*steps] for i in range(4)]
    
    
    # sum_roll, sum_pitch, sum_yaw = np.sum(directions["Roll"]), np.sum(directions["Pitch"]), np.sum(directions["Yaw"])
    # print(sum_pitch, sum_roll, sum_yaw)
    
    for i in range(len(directions["Roll"])): # going through all lists simultaneously
        r, p, y, m = directions["Roll"][i], directions["Pitch"][i], directions["Yaw"][i], magnitudes[i]
        
        val = 0
        
        # check in which range the angle is in
        for j in range(len(ranges)):
            if m >= ranges[j][0] and m < ranges[j][1]: #check if it is in that range
                val = j+1 # this is the range it is in
                
        # then this will be the category and the direction of motion(down/up etc!!)
        # transforming first to misty vals:
        
        full["Roll"].append(linear_mapping(misty_limits, val*r, "Roll"))
        full["Pitch"].append(linear_mapping(misty_limits, val*p, "Pitch"))
        full["Yaw"].append(linear_mapping(misty_limits, val*y, "Yaw"))
    
        
    full["Roll"] = check_turning_points(full["Roll"], indices, threshold=0)
    full["Pitch"] = check_turning_points(full["Pitch"], indices, threshold=0)
    full["Yaw"] = check_turning_points(full["Yaw"], indices, threshold=0)
    

        
    t = motion_dict_nao["time"]
    
    if full["Roll"][1][-1] < t:
        full["Roll"][0].append(0)
        full["Roll"][1].append(t)
    
    if full["Pitch"][1][-1] < t:
        full["Pitch"][0].append(0)
        full["Pitch"][1].append(t)
    
    if full["Yaw"][1][-1] < t:
        full["Yaw"][0].append(0)
        full["Yaw"][1].append(t)
        
    # tot = [directions["Magnitude"][i] * directions["Pitch"][i] for i in range(len(directions["Magnitude"]))]
    # timesteps = np.arange(len(directions['Magnitude']))
    # #time = motion_dict_nao["time"]
    # #s = [pnkt[1] for pnkt in tp]
    # plt.plot(full["Pitch"][1], full["Pitch"][0], color="#1A85FF", label="Misty")
    # plt.plot(timesteps, tot, color="#D41159", label="NAO")
    # plt.title(scene + ": head pitch Misty - z-direction NAO")
    # plt.legend()
    # plt.xlabel("Time")
    # plt.ylabel("Angle (rad)")
    # if scene == "dance_robot":
    #     plt.savefig('/Users/claudiabadescu/Desktop/master2/master latex/images/robot_pitch_cross.png')
    # plt.show()
    

    return full, sig_dirs


# FOR THE TORSO -> driving mapping

def driving(torso_list, scene):
   
    x = [lst[0] for lst in torso_list]
    y = [lst[1] for lst in torso_list]
    
    x_diffs, y_diffs = [], []
    
    for i in range(len(x)-1):
        x_diffs.append(np.abs(x[i]-x[i+1]))
        y_diffs.append(np.abs(y[i]-y[i+1]))
    
    sum_x, sum_y = np.sum(x_diffs), np.sum(y_diffs)  
        
    if sum_x > sum_y: # da er x dominant
        dir = "x"
    else: # da er y dominant
        dir = "y"
        
    plt.plot(x,y, color="#1E88E5")
    plt.title(scene)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
        
    return dir


### ------------------- METHODS FOR BOTH


def check_turning_points(vals, indeces, threshold):
    
    v, idx = [vals[0]], [0]
    
    check_set = set()
    for i in range(1, len(vals)):
        
        if vals[i] != vals[i-1] or (np.abs(vals[i] - vals[i-1]) > threshold): # hvis differansern er større så spara vi på dem
            v.append(vals[i])
            
            new_index = indeces[i]
            while new_index in check_set:
                new_index += 1
            idx.append(new_index)
            check_set.add(new_index)  # Add the new index to the set
           
    #shifted_idx = [idx[i] for i in range(1,len(idx)) if idx[i] != idx[i-1]]
    #shifted_idx.insert(0, 0)
    print("LENTGHS", len(v), len(idx))
    return [v, idx]


def turning_points(l0, l1, l2): # takes in 3 lists
    """finds turning points and returns a list of points/angles back"""
    
    
    turning_points, indeces = [], [] # will be a list of points/angles

    turning_points.append([l0[0], l1[0], l2[0]])
    indeces.append(0)
    
    for i in range(1, len(l0)-1): # hopper over første og går gjennom hele rangen med angles
        
        # må sjekke om det er ekstrempunkt for any of them
        this0, previous0, next0 = l0[i], l0[i-1], l0[i+1] 
        this1, previous1, next1 = l1[i], l1[i-1], l1[i+1] 
        this2, previous2, next2 = l2[i], l2[i-1], l2[i+1] 
        
        # checks:
        l0_check = (this0 > previous0 and this0 > next0) or (this0 < previous0 and this0 < next0) # if true = extreme point
        l1_check = (this1 > previous1 and this1 > next1) or (this1 < previous1 and this1 < next1) 
        l2_check = (this2 > previous2 and this2 > next2) or (this2 < previous2 and this2 < next2) 
        
        if l0_check or l1_check or l2_check: # if any one of them are true, then we have an extreme point.
            turning_points.append([this0, this1, this2]) # appender hele punktet
            indeces.append(round(i/2)) # append the index where the extreme point happened
            
    
    turning_points.append([l0[-1], l1[-1], l2[-1]]) # legger til siste punktet i lista også
    indeces.append(round(len(l0)/2))
        
    return turning_points, indeces
    
            
### ---------------- OTHER:
def get_colors(saturation, brightness, color):
    # https://web.cs.uni-paderborn.de/cgvb/colormaster/web/color-systems/hsv.html
    nr_colors = 10
    hue_space = {"rainbow": [0, 360], "yellow": [0, 60], "green": [60, 120], "teal": [120, 180],
                 "blue": [180, 240], "purple": [240, 300], "pink": [300, 360]}
    
    hue_vals = np.linspace(hue_space[color][0], hue_space[color][1], nr_colors)
    hue_vals = (hue_vals)/360 # normalizing the values to be between 0 and 1
    rgb_vals = []
    for hue in hue_vals: # this is a list of lists
        rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
        rgb_vals.append({"r": float(rgb[0]*255), "g": float(rgb[1]*255), "b": float(rgb[2]*255)})
        
            
    return rgb_vals
    
    
def set_extras(motion_dict_nao, scene, everything):
    
    extra = read_motion("extra.yaml") # loading the eyes/sounds extra stuff
    # adding extra stuff to the yaml
    everything["misty"]["motion_length"] = motion_dict_nao[scene]["remote"]["time"] # getting the length of the motion

    
    # eyes
    everything["misty"]["eyes"] = extra["eyes"][scene][0] # get the first element - the image to be displayed for this motion
        
    # music/sounds
    if scene in extra["songs"].keys():
        everything["misty"]["song"] = extra["songs"][scene] # adding song
        
    # leds:
    if motion_dict_nao[scene]["remote"]["type"] == "gesture": # if gesture, read rgb values and add them to 
        everything["misty"]["leds"] =  extra["leds"][scene] # add the whole dictionary to the new one
        everything["misty"]["song"] = extra["sounds"][scene][0]
        
    else: # then we have dances and we want to calculate the rgb values - based on the intensity of the dances:
        dance_speed = motion_dict_nao[scene]["remote"]["qom"]["speed"]
        everything["misty"]["leds"] = {"qom": {}}
        if dance_speed == "fast": # intense colors
            everything["misty"]["leds"]["qom"]["blinking"] = "true" # want all fast songs to blink
            rgb = get_colors(1.0, 0.70, extra["leds"][scene])
            rate = 600
        else: 
            everything["misty"]["leds"]["qom"]["blinking"] = "false" # slow songs shouldnt blink
            rgb = get_colors(0.45, 1.0, extra["leds"][scene])
            rate = 2500
        
        everything["misty"]["leds"]["qom"]["colors"] = rgb
        everything["misty"]["leds"]["qom"]["rate"] = rate
        
        everything["misty"]["leds"]["direct"] = {"rate": 1100, "blinking": "true"}
        everything["misty"]["leds"]["direct"]["colors"] = get_colors(1.0, 1.0, extra["leds"][scene])
        
        # adding led stuff without qom cals:
        # speed
        everything["misty"]["speed"] = motion_dict_nao[scene]["remote"]["qom"]["speed"] # add if motion is fast or slow
        
    return everything


### ---- QOM STUFF

def find_stats(this_range_vals):

    diff = float(np.max(this_range_vals)-np.min(this_range_vals))
    
    max = float(np.max(this_range_vals))
    min = float(np.min(this_range_vals))
    
    return diff, max, min
    
def average(this_range_vals, max, min):
    avg = float(np.average(this_range_vals))
    
    
    if np.abs(avg-max) < np.abs(avg-min): # then average is closer to max than to min = jevnt høyere verdier in the part = more movement
        intensity = "high"
    else: # then intenisty is lower and average is closer to min
        intensity = "low"
        
    return intensity, avg

def find_fluctuations(intensity, diff, avg):
    
    diff_diff_avg = np.abs(diff-avg)
    fluctuations = False

    if diff_diff_avg > 0.0001: # the difference is significant
        if (intensity == "low" and diff >= avg) or (intensity == "high" and diff <= avg): # then lots of fluctutations
            fluctuations = True
            

    
    return fluctuations


def get_segment_info(qom_vals, ext):

   #ext = ext[1:]
    
    parts = {}
    
    for i in range(len(ext)-1): # each part
        part = {}
        
        part["range"] = [ceil(ext[i]/30), ceil(ext[i+1]/30)] # adding the range indexes

        this_range_vals = qom_vals[ext[i]:ext[i+1]]
            
        #part["elements"] = find_nr_elements(idxs, part["range"])
            
        diff, max, min = find_stats(this_range_vals)
            
        intensity, avg = average(this_range_vals, max, min)
        fluctuations = find_fluctuations(intensity, diff, avg)
            
        part["fluctuations"] = fluctuations 
        part["intensity"] = intensity
            
        parts[f"part{i}"] =  part
        
    return parts


def append_angles(resulting_angles, joint, intensity, upper, lower, choose):
    # appending angles based on intensity
    
    # setting vals based on intensity
    if intensity == "high": 
        val1, val2 = float(random.uniform(*upper)), float(random.uniform(*lower))
        resulting_angles[joint].append(val1)
        resulting_angles[joint].append(val2)
                
    else: # low intensity
        val1, val2 = float(random.uniform(*choose)), float(random.uniform(*choose))
        resulting_angles[joint].append(val1)
        resulting_angles[joint].append(val2)
    
    return resulting_angles, val1, val2

def get_ranges(joint, misty_limits):    
    limits_list = misty_limits[joint] # getting the arm limits
    
    three = (np.abs(limits_list[0]) + limits_list[1])/3
    
    # dividing the motion into 3 ranges:
    upper_range = [limits_list[0], limits_list[0]+three]
    middle_range = [upper_range[1], upper_range[1]+three]
    lower_range = [middle_range[1], limits_list[1]]  
    
    return upper_range, middle_range, lower_range

def get_new_intensity(fluctuations, intensity):
    
    if fluctuations == "true":
        if intensity == "high":
            intensity = "low"
        elif intensity == "low":
            intensity == "high"
            
    return intensity

def angle_dances_hands(qom, ext, misty_limits):
    # gjør dette for hver scene
    
    part_info = get_segment_info(qom, ext)
    
    upper_range, middle_range, lower_range = get_ranges("LArm", misty_limits)
    
    resulting_angles = {"LArm": [], "RArm": []}

    for i in range(len(part_info)): # going through all the parts and getting info
        
        intensity = part_info[f"part{i}"]["intensity"]
        fluctuations = part_info[f"part{i}"]["fluctuations"]
        rng = part_info[f"part{i}"]["range"] # getting a list over the range
        
        # for each part we choose whether the arms should move at the same time or not.
        same = random.choice(["same", "diff"]) # choose randomly if the arms should move at the same time or not
        # adding vals to the finar list
        for val in range(rng[0], rng[1],2):
            print(val)
            # then we have a lot of fluctuations - add two vals - two from high and two from low
            choose_range = random.choice([upper_range, middle_range, lower_range])
            
            resulting_angles, val1, val2 = append_angles(resulting_angles, "LArm", intensity, upper_range, lower_range, choose_range)
            
                    
            if same == "same": #both hands get the same value
                    resulting_angles["RArm"].append(val1)
                    resulting_angles["RArm"].append(val2)
                    
            else: # not same hands - right hand gets new values
                if intensity == "high":
                    val1_new, val2_new = float(random.uniform(*upper_range)), float(random.uniform(*lower_range))
                else:
                    val1_new, val2_new = float(random.uniform(*choose_range)), float(random.uniform(*choose_range))
                      
                resulting_angles["RArm"].append(val1_new)
                resulting_angles["RArm"].append(val2_new)
                    
            
            # change intensity between each iteration if the fluctuations are true, and if not, dont change it
            intensity = get_new_intensity(fluctuations, intensity)
        print("NEW ")   
    return resulting_angles # commands in rad to be sent to misty for the hands
            
          

    
def angle_dances_head(qom, ext, misty_limits):
    
    part_info = get_segment_info(qom, ext)
    
    ranges_dict = {"Roll": get_ranges("Roll", misty_limits), "Pitch": get_ranges("Pitch", misty_limits),
                   "Yaw": get_ranges("Yaw", misty_limits)}
    
    resulting_angles = {"Yaw": [], "Pitch": [], "Roll": []}
    
    # her slipper e å tenke på same/ikke same
    choices = ["Roll", "Pitch", "Yaw"]
   
    
    for i in range(len(part_info)): # going through all the parts and getting info
        
        intensity = part_info[f"part{i}"]["intensity"]
        fluctuations = part_info[f"part{i}"]["fluctuations"]
        rng = part_info[f"part{i}"]["range"] # getting a list over the range
        
        print("RAAANGES:", rng, intensity)
        
        # choosing 1 dir
        dir = random.choice(choices)
        rngs = ranges_dict[dir]
        
        choices.remove(dir)
        
        if intensity == "high":
            # choosing two directions

            dir2 = random.choice(choices)
            choices.remove(dir2)
            rngs2 = ranges_dict[dir2]
        
        # adding vals to the finar list
        for val in range(rng[0], rng[1],2):
            
            resulting_angles, _, _ = append_angles(resulting_angles, dir, intensity, rngs[0], rngs[1], rngs[2])
            
            if intensity == "high":
                resulting_angles, _, _ = append_angles(resulting_angles, dir2, intensity, rngs2[0], rngs2[1], rngs2[2])

            # adding 0s for the directions where there is no movement:
            for remaining_dirs in choices:
                resulting_angles[remaining_dirs].append(0.0)
                resulting_angles[remaining_dirs].append(0.0)
    
            intensity = get_new_intensity(fluctuations, intensity)
            
        choices = ["Roll", "Pitch", "Yaw"] # resetting the choices list
        
    return resulting_angles
 
 
def determine_driving():
    dir = random.choice(["x", "y"])
    return dir
    

 
#### -----  WRITE TO YAML

def write_to_yaml(filename, data, scene):
            
    # antar at vi allerede har en fil og at den er fylt - siden vi laga den i nao fila
    with open(filename, 'r') as file:
        content = yaml.safe_load(file) # loading old content

        if content.get(scene):
            content[scene].update(data) # update scene data
    
    with open(filename, 'w') as file:
        yaml.safe_dump(content, file, default_flow_style=False)

def scale_arms(misty_limits, arm_dict):
    min_larm = np.min(arm_dict["LArm"][0])
    max_larm = np.max(arm_dict["LArm"][0])
    
    min_rarm = np.min(arm_dict["RArm"][0])
    max_rarm = np.max(arm_dict["RArm"][0])

    new_l, new_r = [], []
    
    for i in range(len(arm_dict["LArm"][0])):
        if max_larm-min_larm == 0:
            formula_l = 0.0
        else:
            formula_l = ((arm_dict["LArm"][0][i]- min_larm)/(max_larm-min_larm)) * (misty_limits["LArm"][1]-misty_limits["LArm"][0]) + misty_limits["LArm"][0]
        new_l.append(float(formula_l))
        
    for i in range(len(arm_dict["RArm"][0])):
        if max_rarm-min_rarm == 0:
            formula_r = 0.0
        else:
            formula_r = ((arm_dict["RArm"][0][i]- min_rarm)/(max_rarm-min_rarm)) * (misty_limits["RArm"][1]-misty_limits["RArm"][0]) + misty_limits["RArm"][0]
        new_r.append(float(formula_r))
        
        
    new = {"LArm": [new_l, arm_dict["LArm"][1]], "RArm": [new_r, arm_dict["RArm"][1]]}
            
    return new


def assemble_and_write(filepath):
    
    save = {}
    
    handler = "remote"
    
    motion_dict_nao = read_motion("nao_output.yaml")  
    nao_limits = read_csv("robot_limits/nao_limits.txt")
    misty_limits = read_csv("robot_limits/misty_limits.txt")

    
    #speed(motion_dict_nao, "robot", 70)
    
    #mappings_hands = {"LArm": "HeadPitch", "RArm": "HeadPitch"} # defining the mappings for the Misty -> Nao
    
    
    for scene in motion_dict_nao: 
        print(scene)
        # må sjekke taggene til bevegelsen
        
        everything = {"misty": {}}
        
        if motion_dict_nao[scene][handler]["special"]: # if true - dont map normally
            continue
        
        motion_dict_joints = motion_dict_nao[scene]["remote"]["joint_angles"] # angles
        #motion_dict_xyz = motion_dict_nao[scene]["remote"]["endeffector_positions"]["LArm"] # xyz
        motion_dict_xyz = motion_dict_nao[scene]["remote"]
        
        # if "qom" not in motion_dict_nao[scene]["remote"].keys():
        #     continue
        
            
        arms, head = {}, {}
             
            
        if motion_dict_nao[scene][handler]["type"] == "gesture": # then we have mapping - head = head, hands = hands
            
            nao_limits["LShoulderPitch"][0] = 0
            nao_limits["RShoulderPitch"][0] = 0
            nao_limits["LShoulderPitch"][1] = 1.57
            nao_limits["RShoulderPitch"][1] = 1.57
            
            arms_mapping = {"LArm": "LShoulderPitch", "RArm": "RShoulderPitch"} 
            # if scene == "greet": # want another mapping for greet
            #     arms_mapping = {"LArm": "LElbowRoll", "RArm": "RElbowRoll"}
            
            head_mapping = {"Yaw": "HeadYaw", "Pitch": "HeadPitch"}
            
            head = processing_angles(nao_limits, misty_limits, motion_dict_joints, scene, head_mapping, threshold=0.1)
            arms = processing_angles(nao_limits, misty_limits, motion_dict_joints, scene, arms_mapping, threshold=0.1)
            #arms = map_angles_misty(arms_mapping, motion_dict_joints, scene, misty_limits, nao_limits)
            #head = map_angles_misty(head_mapping, motion_dict_joints, scene, misty_limits, nao_limits)
            new_arms = scale_arms(misty_limits, arms)
            if scene == "raise_hand":
                arms = {"LArm": [[-0.4887],[0]], "RArm": [[1.2217], [0]]}
   
            #head["Roll"] = [[0.0],[0]]
            everything["misty"] = {"direct": {"arms": arms, "head": head}}
                
        else: # if its a dance though we use mapping: head -> hands (xyz) 
            
            wheels = driving(motion_dict_xyz["endeffector_positions"]["Torso"], scene)
           
            # option 1: hands -> head (xyz), head -> hands (joint) - tested
            misty_xyz_head, sig_dirs = xyz(motion_dict_xyz, scene, misty_limits) # using x axis too
            arms_mapping = {"LArm": "LShoulderPitch", "RArm": "RShoulderPitch"}  # can also be head  (???)
            #arms = map_angles_misty(arms_mapping, motion_dict_joints, scene, misty_limits, nao_limits)
            arms = processing_angles(nao_limits, misty_limits, motion_dict_joints, scene, arms_mapping, threshold=0.2)
            new_arms = scale_arms(misty_limits, arms)
            # constructing the assembled dictionary that we will write to file:
            everything["misty"]["cross"] = {"arms": new_arms, "head": misty_xyz_head, "wheels" : wheels} # constructing 
            everything["misty"]["cross"]["head"]["directions"] = sig_dirs #legg til sig

            # option 3
            head_mapping = {"Yaw": "HeadYaw", "Pitch": "HeadPitch"}
            arms_mapping = {"LArm": "LShoulderPitch", "RArm": "RShoulderPitch"}  # can also be head  (???)
            #arms_mapping = {"LArm": "LShoulderPitch", "RArm": "RShoulderPitch"} 
            head = processing_angles(nao_limits, misty_limits, motion_dict_joints, scene, head_mapping, threshold=0.2)
            arms = processing_angles(nao_limits, misty_limits, motion_dict_joints, scene, arms_mapping, threshold=0.2)
            new_arms = scale_arms(misty_limits, arms)
            
            
            everything["misty"]["direct"] = {"arms": new_arms, "head": head, "wheels" : wheels}
            
            
            # setting qom stuff too - for dances only
            qom_vals = motion_dict_nao[scene]["remote"]["qom"]
                        
            arm_angles = angle_dances_hands(qom_vals["qom_vals"][30:-30], qom_vals["extreme_points"], misty_limits)
            head_angles = angle_dances_head(qom_vals["qom_vals"][30:-30], qom_vals["extreme_points"], misty_limits)
            dir = determine_driving()
            
            everything["misty"]["qom"] = {"arms": arm_angles, "head": head_angles, "wheels": dir}
                

        everything = set_extras(motion_dict_nao, scene, everything)
        
        

        # save to yaml    
        filename = filepath+scene+".yaml"
        #write_to_yaml(filename, everything, scene)


if __name__ == "__main__":
    filepath = "/Users/claudiabadescu/Desktop/master2/ztl_remote/tasks/" 
    assemble_and_write(filepath)