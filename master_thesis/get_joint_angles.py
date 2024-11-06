#!/usr/bin/python2

import qi
import sys
import time
import motion
import yaml
import os


# using the api for help: http://doc.aldebaran.com/2-8/naoqi/motion/control-joint-api.html#ALMotionProxy::getAngles__AL::ALValueCR.bCR

# kinematic chains: http://doc.aldebaran.com/2-8/family/nao_technical/bodyparts_naov6.html#nao-chains 

# In this script we connect to the robot, and read of the angle values while the robot moves and we append them to an output script. 
# The idea is that we start this script and we estimate the duration of the movement and then we start the movement from the nao remote control and we
# record the movement to file while nao actually moves.

# it also gets the position of the hand in space

# how to run: python2 get_joint_angles.py *duration* *movement*

def connect_to_robot(ip, port):
    session = qi.Session()

    try: 
        session.connect("tcp://" + ip + ":" + str(port))
    except RuntimeError:
        print("Cannot connect to Naoqi at this ip and on this port.")
        sys.exit(1)
        
    return session

def read_values(session):

    motion_service = session.service("ALMotion")
    relevant_joint_names = "Body" # want all angles to get recorded and on a loop for the entire motion
    relevant_points = ["LArm", "RArm", "Torso"] # only tracking the hands
    
    frame_position = motion.FRAME_TORSO
    frame_orientation = motion.FRAME_WORLD

    # using this frame cause it provides a "natural ego-centric reference" and it gives a consistent
    # frame that doesnt depend on the movement of the torso

    duration = sys.argv[1]

    use_sensors = False
    all_angle_readings = []
    all_position_readings = []
    
    t = time.time() # tid ved start
    interval = 0.2
    
    while (time.time() - t) <= (int(duration)+2):
        
        iteration_start = time.time()
        
        get_joint_angles = motion_service.getAngles(relevant_joint_names, use_sensors) # getting the commanded angle values (rad) for each joint

        get_positions1 = motion_service.getPosition(relevant_points[0], frame_position, use_sensors) # returns 6vec with pos (m) and rot (rad) -  (x, y, z, wx, wy, wz)
        get_positions2 = motion_service.getPosition(relevant_points[1], frame_position, use_sensors)
        get_orientation = motion_service.getPosition(relevant_points[2], frame_orientation, use_sensors) # getting torso in 6vec with pos and rot
     
        all_angle_readings.append(get_joint_angles)
        all_position_readings.append([get_positions1, get_positions2, get_orientation])

        iteration_time = time.time()-iteration_start
        
        time_sleep = interval-iteration_time
        if time_sleep > 0:
            time.sleep(time_sleep)
       
       
    return all_angle_readings, all_position_readings, motion_service, relevant_points
        

def sorting(name_list, value_list):
    """This method takes a list of body names from the robot, creates a dictionary 
    and organises through them so that the dictionary finally has the form: 
    dict = {"name1":[t1, t2, t2...], "name2":[t1, t2, t2...],...}, where t1 and so on
    corresponds to the value of the joint joint/position etc at different time stamps."""

    values_dict = {name:[] for name in name_list} # creating a dictionary for all joints where each joint has a list

    for values in value_list: # iterating through all recorded values
        for name, value in zip(name_list, values):
            values_dict[name].append(value) 

    return values_dict


def nao_task_yaml(scene):
    """this method writes to the task yaml with nao information - cause the file name is also the button that is pressed 
    and command that is sent..."""

    task_path = "/Users/claudiabadescu/Desktop/Robotikkmaster/MASTEROPPGAVE/master/ztl_remote/tasks/"
    filename = task_path + scene + ".yaml"
    
    content = {scene: {"nao": scene}}
    
    # skriv til task yaml file for nao handler
    with open(filename, 'w') as file:
        yaml.safe_dump(content, file, default_flow_style=False)


def cleanup(nested): # nested list in
    

    cleaned = [nested[0]] # adding first list in list
    indeces_saved = []
    
    for i in range(len(nested)-3): # going though all lists og finner
        this, next, nextnext, nextnextnext = nested[i], nested[i+1], nested[i+2], nested[i+3]
        
        check = []
        
        for j in range(len(this)): # going through all values
            # if its over threshold it is moving - this is 2 seconds
            #and (abs(next[j]-nextnext[j]) > 0.005) and (abs(nextnext[j]-nextnextnext[j]) > 0.005)
            if (abs(this[j]-next[j]) > 0.002):
                check.append(True) 
        
                # if all values in check = True then we are moving
            
        if len(check) != 0: # if check = True, we are moving and we can append this reading to the list
            cleaned.append(this) 
            indeces_saved.append(i)
            print("hihihih")
        else:
            print("ZEROZEROZERO")
            
    cleaned.append(nested[-1]) # for good measure
     
    return cleaned, indeces_saved


def scene_tag(scene, handler, scenes, time):
    # marker det som en dans viss det er det navnet

    if "dance" in scene:
        scenes[scene][handler]["type"] = "dance"
    else: # then its not a dance but a gesture
        scenes[scene][handler]["type"] = "gesture"
        
    if scene == "rest" or scene == "wake_up": # markere det som en nao scene og ikke for alle robotene - krever spesiell behandling
        scenes[scene][handler]["special"] = True
    else: 
        scenes[scene][handler]["special"] = False
        
    scenes[scene][handler]["time"] = time
        
    return scenes

# ------
### MAIN AND YAML

def write_yaml(joint_angles, endeffector_positions, time):
    """This function is supposed to write all this info into a yaml file. Can write stuff in dictionaries."""
    
    handler = "remote"
    
    #components = {} # dicitonary of components - {component1: info, component2: info ...}

    # # read from file what kinds of actions/buttons from remote we have that we are going to write to our yaml file
    # with open(remote_button_file, "r") as file:
    #     actions = [line.strip() for line in file]
    
    scene = sys.argv[2] # this will tell us the scene name because that is the task we record
    scenes = {scene: {handler: {}}}
    
    # writing to task file:
    nao_task_yaml(scene)
    
    # appending each component into scenes.
    scenes[scene][handler]["joint_angles"] = joint_angles # adding a new key - which is also a dicitionary - edning in 4 levels
    scenes[scene][handler]["endeffector_positions"] = endeffector_positions
    
    filename = "nao_output.yaml"
    
    scenes = scene_tag(scene, handler, scenes, time)
    
    if os.path.isfile(filename): # check if it exists, you dont have to read from it if it does
        with open(filename, 'r') as file:
            content = yaml.safe_load(file) # loading old content
            
        content.update(scenes) # updating yaml data
    else:
        content = scenes
        
    with open(filename, 'w') as file:
        yaml.safe_dump(content, file, default_flow_style=False)
    
def main():
    
    # this writes 
    
    # start session 
    session = connect_to_robot("10.0.1.14", 9559)   
    all_angle_readings, all_position_readings, motion_service, relevant_points = read_values(session)
    
    cleaned_angles, indices_saved = cleanup(all_angle_readings)
    time_taken = int(round(len(indices_saved)/2))


    cleaned_positions = [all_position_readings[i] for i in range(len(all_position_readings)) if i in indices_saved]

    all_body_names = motion_service.getBodyNames("Body") # to get the names of all joints in the body
    joint_angles = sorting(all_body_names, cleaned_angles)
    endeffector_positions = sorting(relevant_points, cleaned_positions)
    
    write_yaml(joint_angles, endeffector_positions, time_taken)
    

main()
       
    
