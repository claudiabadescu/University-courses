from ztl.core.task import ExecutableTask
from mistyPy.Robot import Robot
import base64


import websocket
import json
import threading
import ast
import numpy as np
import time
import random


# websocket: https://websocket-client.readthedocs.io/en/latest/examples.html#readme-examples

# opp til misty å decode meldinga

class Misty(ExecutableTask): # actual task logic

  def __init__(self, payload):
    self.active = True
    #self.duration = duration
    self.payload = payload
    
    self.ip = "172.20.10.13"
    self.misty = Robot(self.ip) ### the IP address should come from the server i think!!!
    self.reached_y, self.reached_r, self.reached_p = False, False, False
    self.angle_y, self.angle_r, self.angle_p = 0.0, 0.0, 0.0
    self.reached_pos = threading.Event() # chatgpts idea - creating an event 
    
    self.reached_right, self.reached_left = False, False
    self.angle_right, self.angle_left = 0.0, 0.0
    self.reached_hand_pos = threading.Event()
    
    
    self.motion_done = threading.Event() # new event for when the motion is done
    

    
  def initialise(self):
    return True


  def on_message(self, ws, message):
    #check if the head is where it should be
    received_msg = json.loads(message)
    actuator = received_msg["message"]["sensorId"] # getting the actuator type
    value = np.deg2rad(received_msg["message"]["value"])
    print(actuator, value)
    print(self.angle_r, self.angle_y, self.angle_p)
    
    threshold = 0.15
    
    if actuator == "ahy" and (self.angle_y-threshold <= value <= self.angle_y+threshold):
      self.reached_y = True
    if actuator == "ahr" and (self.angle_r-threshold <= value <= self.angle_r+threshold):
      self.reached_r = True
    if actuator == "ahp" and (self.angle_p-threshold <= value <= self.angle_p+threshold):
      self.reached_p = True
    
    # loo
    if self.reached_y and self.reached_p and self.reached_r: # if all are true
      self.reached_pos.set() # then we have reached the desired pos
      
    if actuator == "ala" and (self.angle_left-threshold <= value <= self.angle_left+threshold):
      self.reached_left = True
    
    if actuator == "ara" and (self.angle_right-threshold <= value <= self.angle_right+threshold):
      self.reached_right = True
    
    if self.reached_left  and self.reached_right: # if all are true
      self.reached_hand_pos.set() # then we have reached the desired pos
      
    

  def on_error(self, ws, error):
    print(error)

  def on_close(self, ws):
    unsubscribe_msg = {
    "Operation": "unsubscribe",
    "EventName": "read_actuators_new",
    }
    msg = json.dumps(unsubscribe_msg)
    self.ws.send(msg)


  def on_open(self, ws):
    """This method subscribes to the published stuff."""
    
    print("Opened connection")
    
    subscribe_msg_yaw = {    
      "Operation": "subscribe", # create subscription
      "Type": "ActuatorPosition", # event type to subscribe to
      "DebounceMs": 100, # send data every 100 milliseconds            
      "EventName": "ap1", #name of this subscription
      "EventConditions": [{"Property": "sensorId", "sensorId": "ahy", "Inequality": "=", "Value": "ahy"}]  
    }
    
    subscribe_msg_roll = {
            
      "Operation": "subscribe", # create subscription
      "Type": "ActuatorPosition", # event type to subscribe to
      "DebounceMs": 100, # send data every 100 milliseconds            
      "EventName": "ap2", #name of this subscription
      "EventConditions": [{"Property": "sensorId", "sensorId": "ahr", "Inequality": "=", "Value": "ahr"}] 
    }
    
    subscribe_msg_pitch = {
            
      "Operation": "subscribe", # create subscription
      "Type": "ActuatorPosition", # event type to subscribe to
      "DebounceMs": 100, # send data every 100 milliseconds            
      "EventName": "ap3", #name of this subscription
      "EventConditions": [{"Property": "sensorId", "sensorId": "ahp", "Inequality": "=", "Value": "ahp"}] 
    }
    
    subscribe_msg_left = {
            
      "Operation": "subscribe", # create subscription
      "Type": "ActuatorPosition", # event type to subscribe to
      "DebounceMs": 100, # send data every 100 milliseconds            
      "EventName": "ap4", #name of this subscription
      "EventConditions": [{"Property": "sensorId", "sensorId": "ala", "Inequality": "=", "Value": "ala"}] 
    }
    
    subscribe_msg_right = {
            
      "Operation": "subscribe", # create subscription
      "Type": "ActuatorPosition", # event type to subscribe to
      "DebounceMs": 100, # send data every 100 milliseconds            
      "EventName": "ap5", #name of this subscription
      "EventConditions": [{"Property": "sensorId", "sensorId": "ara", "Inequality": "=", "Value": "ara"}] 
    }
   
    print("hæææ")
    msg_y = json.dumps(subscribe_msg_yaw)
    msg_r = json.dumps(subscribe_msg_roll)
    msg_p = json.dumps(subscribe_msg_pitch)
    msg_right = json.dumps(subscribe_msg_left)
    msg_left = json.dumps(subscribe_msg_right)

    print("hmm")
    self.ws.send(msg_y)
    self.ws.send(msg_r)
    self.ws.send(msg_p)
    self.ws.send(msg_right)
    self.ws.send(msg_left)

    print("sent???")
    
    
  ########## DIRECT/COMBO MAPPING:
  
  def find_dir(self, dirs, i, motion_length, ny_j, previous, next_max):
    # use this for when you CROSS MAP stuff - DANCES!!!!
    # find in which range this i lies in for the directions:
  
    for j in range(len(dirs)): # go through all intervals
      if i >= dirs[j][-1][0] and i < dirs[j][-1][1]:
        # then i is in the jth interval
        # get the x, y, z vals for this interval
        #xyz_max = max(dirs[j][0], key=dirs[j][0].get)
        _, xyz_max = zip(*sorted(zip(dirs[j][0].values(), dirs[j][0].keys()))) # sorting directions
        xyz_max = list(xyz_max)
                
        if j != ny_j: # if we have changed part of the dance
          if xyz_max[-1] == previous: # check if new direcion is the same as old direction
            random_dir = random.randint(0,1)
            dir = xyz_max[random_dir]  # choosing next directory at random
            next_max = True
          else: # if the new direction is different from the previous
            dir = xyz_max[-1]
            next_max = False
            ny_j = j
            previous = dir
        else: # if we have not changed dance segment
          if next_max:
            dir = xyz_max[1]
          else: 
            dir = xyz_max[-1]
      elif i == (motion_length-1) or i == motion_length:
        dir = ""
    
    return dir, ny_j, previous, next_max

  def read_from_yaml(self, goal_dict, map, mapping, scene_name):
    # getting the info for each joint from yaml
    
    if map == "combo" and mapping == "direct":
      map = "direct"
    
    elif map == "combo" and mapping == "cross":
      print("yeyyy")
      map = "cross"
      
    
    rarm = goal_dict[map]["arms"]["RArm"]
    larm = goal_dict[map]["arms"]["LArm"]
    pitch = goal_dict[map]["head"]["Pitch"]
    roll = goal_dict[map]["head"]["Roll"]
    yaw = goal_dict[map]["head"]["Yaw"]
    
    if map == "qom" and "dance" in scene_name:
      pitch_dict, roll_dict, yaw_dict, rarm_dict, larm_dict = pitch, roll, yaw, rarm, larm
    else:
      # making dictionaries - for the timing
      pitch_dict = {index: value for index, value in zip(pitch[1], pitch[0])}
      roll_dict = {index: value for index, value in zip(roll[1], roll[0])}
      yaw_dict = {index: value for index, value in zip(yaw[1], yaw[0])}
      rarm_dict = {index: value for index, value in zip(rarm[1], rarm[0])}
      larm_dict = {index: value for index, value in zip(larm[1], larm[0])}
    
    
    if "dance" in scene_name:
      wheels = goal_dict[map]["wheels"]
      speed = goal_dict["speed"]
    else:
      wheels, speed = "", ""
      
      
    
    motion_length = goal_dict["motion_length"]
    
    song = goal_dict["song"]
  
    eyes = goal_dict["eyes"]
    
    leds = goal_dict["leds"]
    
    return pitch_dict, roll_dict, yaw_dict, rarm_dict, larm_dict, wheels, motion_length, song, speed, eyes, leds
    
  
  def start_drive(self, motion_length, dir, speed):
    turn = False
    
    if dir == "y": # if y dir
      lin_vel = 0
      start_time = 2.5
      if speed == "fast":
        t = 3.5
        ang_vel = 90 # if fast
      elif speed == "slow":
        t = 5
        ang_vel = 5 # if slow
      else: # we dont have speed mapping
        ang_vel = 50
        t = 4
      
    else: # if x dir
      ang_vel = 0
      start_time = 0.65
      if speed == "fast":
        t = 0.65
        lin_vel = 15
      elif speed == "slow":
        lin_vel = 3
        t = 0.85
      else:
        lin_vel = 10
        t = 0.85
      

    self.misty.drive_time(linearVelocity=lin_vel, angularVelocity=ang_vel, timeMs=start_time*1000)
    time.sleep(start_time+1)
    
    #for i in range(motion_length):
    
    while True:
        
      if turn == False:
        self.misty.drive_time(linearVelocity=-lin_vel, angularVelocity=-ang_vel, timeMs=t*1000)
        print("TURN LEFT")
        turn = True
      else: # turn is true
        self.misty.drive_time(linearVelocity=lin_vel, angularVelocity=ang_vel, timeMs=t*1000)
        turn = False
        print("TURN RIGHT")
        
      if self.motion_done.is_set(): # hvis flagget er satt, exit
        print("flagget er satt")
        break
    
      
      time.sleep(t+1)
    
    if turn == False:
      self.misty.drive_time(linearVelocity=lin_vel, angularVelocity=ang_vel, timeMs=start_time*1000)
    else: 
      self.misty.drive_time(linearVelocity=-lin_vel, angularVelocity=-ang_vel, timeMs=start_time*1000)



  def set_extras(self, leds, eyes, method, scene_name):
    self.misty.display_image(eyes)
    
    # same behavior of leds for qom and combo
    
    if "dance" in scene_name:
      if method == "combo":
        method = "qom"
      
      leds = leds[method] #led behavior depends on if we are using only direct or qom/combo
      
      if leds["blinking"] == "true":
        trans_type = "blink"
      else:
        trans_type = "breathe"
        
      time_between = leds["rate"]
      
      while True: # while flag is not set set - display leds
        
        if self.motion_done.is_set():
          break
        
        for i in range(len(leds["colors"])-1): # going through all of them and transition between first and last
          this, next = leds["colors"][i], leds["colors"][i+1]
          
          self.misty.transition_led(this["r"], this["g"], this["b"], next["r"], next["g"], next["b"], transitionType=trans_type, timeMs=time_between)
        
          time.sleep(time_between/1000)
      
    else:
      while True: # while flag is not set set - display leds
        
        if self.motion_done.is_set():
          break
        
        self.misty.change_led(red=leds["r"], green=leds["g"], blue=leds["g"])
      
      
    # reset tilbake etter 2 sek
    time.sleep(2)
    self.misty.change_led(100, 70, 160)
    self.misty.display_image("e_DefaultContent.jpg")
    
      

    
  def move_direct(self, motion_length, pitch_dict, roll_dict, yaw_dict, rarm_dict, larm_dict, wheels, speed, eyes, leds, song, mapping, method, scene_name, dirs=""):
    """Method that sends commands to misty for the mapping hands -> head xyz:"""
    
    # this is info i get from qom!!      
    if speed == "slow":
      head_vel = 85
      arms_vel = 40
    elif speed == "fast": # fast/ikkeno
      head_vel = 95
      arms_vel = 75
    else:
      head_vel = 80
      arms_vel = 70

      
    if "dance" not in scene_name: 
      head_vel = 90
      arms_vel = 60
    
    # # init values for all of them and only change when idx changes
    ap, aro, ay, ar, al = pitch_dict[0], roll_dict[0], yaw_dict[0], rarm_dict[0], larm_dict[0]
    
    self.misty.play_audio(song, 5)
    
    extra_thread = threading.Thread(target=self.set_extras, args=[leds, eyes, method, scene_name])
    extra_thread.start()
    
    # arm_thread = threading.Thread(target=self.move_arms_direct, args=[motion_length, rarm_dict, larm_dict, arms_vel])
    # arm_thread.start()
    
    if "dance" in scene_name:
      # sending driving commands in a separate thread
      drive_thread = threading.Thread(target=self.start_drive, args=[motion_length,wheels, speed])
      drive_thread.start()
      
    ny_j = 0
    previous = None
    next_max = False
    
    for i in range(motion_length+1):
      print("i",i)
      # if mapping is cross, then we find significant direction (cross == xyz position method) 
      dir = ""
     
      if mapping == "cross":
        print("HALLO")
    
        dir, ny_j, previous, next_max = self.find_dir(dirs, i, motion_length, ny_j, previous, next_max) 
          
        print("DIR: ", dir)
   
        
      # Use dictionary.get() which returns None if key is not found, else returns the value
      # updating only if new value
      ap = pitch_dict.get(i, ap)
      aro = roll_dict.get(i, aro)
      ay = yaw_dict.get(i, ay)
      ar = rarm_dict.get(i, ar)
      al = larm_dict.get(i, al)
      
      print(np.rad2deg(ap), np.rad2deg(aro), np.rad2deg(ay), np.rad2deg(ar), np.rad2deg(al))
           
      if dir == "Pitch":
        self.angle_y, self.angle_r, self.angle_p  = 0.0, 0.0, ap
        self.misty.move_head(yaw=0, pitch=ap, roll=0, units="radians", velocity=head_vel)
            
      elif dir == "Roll":
        self.angle_y, self.angle_r, self.angle_p  = 0.0, aro, 0.0
        self.misty.move_head(yaw=0, pitch=0, roll=aro, units="radians", velocity=head_vel)
            
      elif dir == "Yaw":
        self.angle_y, self.angle_r, self.angle_p  = ay, 0.0, 0.0
        self.misty.move_head(yaw=ay, pitch=0, roll=0, units="radians", velocity=head_vel)
      
      else:
        self.angle_y, self.angle_r, self.angle_p  = ay, 0.0, ap
        self.misty.move_head(yaw=ay, pitch=ap, roll=0, units="radians", velocity=head_vel) 
        
      self.angle_right, self.angle_left = ar, al
      self.misty.move_arms(rightArmPosition=ar, leftArmPosition=al, leftArmVelocity=arms_vel, rightArmVelocity=arms_vel, units="radians")
      
      self.reached_hand_pos.wait()
      self.reached_left, self.reached_right = False, False
      self.reached_hand_pos.clear()
              
      self.reached_pos.wait() # wait here until the head angles match
      self.reached_y, self.reached_r, self.reached_p = False, False, False # reset
      #self.angle_y, self.angle_r, self.angle_p  = 0.0, 0.0, 0.0
      self.reached_pos.clear() # clear it so it can be set again
      
    #self.angle_y, self.angle_r, self.angle_p  = 0.0, 0.0, 0.0
    self.misty.move_head(yaw=0, pitch=0, roll=0, units="radians", velocity=head_vel)
    self.misty.move_arms(rightArmPosition=70, leftArmPosition=70, leftArmVelocity=arms_vel, rightArmVelocity=arms_vel, units="degrees")

    self.reached_pos.wait() # wait here until the head angles match
    
    self.reached_y, self.reached_r, self.reached_p = False, False, False # reset
    self.reached_pos.clear() # clear it so it can be set again
    self.reached_left, self.reached_right = False, False
    self.reached_hand_pos.clear()
    
    self.motion_done.set() # when this is set - driving should stop
    
    self.ws.close()
    
    
    
  ########## QOM:
  
  def start_movement_thread(self, method, args):
    ws_thread = threading.Thread(target=method, args=args)
    ws_thread.start()
    
      
  def head_qom(self, speed_head, pitch_dict, roll_dict, yaw_dict, rarm_dict, larm_dict, speed_hands):
    
    for i in range(len(pitch_dict)):
      
      self.misty.move_arms(rightArmPosition=rarm_dict[i], leftArmPosition=larm_dict[i], leftArmVelocity=speed_hands, rightArmVelocity=speed_hands, units="radians")

      self.angle_y, self.angle_r, self.angle_p  = yaw_dict[i], roll_dict[i], pitch_dict[i] 
      self.misty.move_head(pitch=pitch_dict[i], roll=roll_dict[i], yaw=yaw_dict[i], velocity=speed_head, units="radians")
      
      # wait until position is reached before continuing
      self.reached_pos.wait() # wait here until the head angles match
      self.reached_y, self.reached_r, self.reached_p = False, False, False # reset
      self.reached_pos.clear() # clear it so it can be set again
      
    print()
    self.angle_y, self.angle_r, self.angle_p  = 0.0, 0.0, 0.0
    self.misty.move_head(yaw=0, pitch=0, roll=0, units="degrees", velocity=speed_head)

    self.motion_done.set() # set motion done flag cause we are done now
    
      

  def move_intensity(self, motion_length, pitch_dict, roll_dict, yaw_dict, rarm_dict, larm_dict, wheels, speed, eyes, leds, song, scene_name):
    """ method to move things ut ifra intensity value"""
    
    if speed == "slow":
      speed_head = 85
      speed_hands = 40
    else: # fast/ikkeno
      speed_head = 95
      speed_hands = 75
      
    self.misty.play_audio(song, 5)
    
    #self.start_movement_thread(self.arms_qom, [speed_hands, rarm_dict, larm_dict])
    self.start_movement_thread(self.head_qom, [speed_head, pitch_dict, roll_dict, yaw_dict, rarm_dict, larm_dict, speed_hands])
    self.start_movement_thread(self.start_drive, [motion_length, wheels, speed])
    self.start_movement_thread(self.set_extras, [leds, eyes, "qom", scene_name])
      
    # når vi har kommet hit har motion flagget blitt satt så vi må resette før neste kjøring 
    # her må e vente til alle motionsa e ferdig!!!
    self.motion_done.wait() # vente på flagget      
    print("DONE")
    self.misty.move_arms(rightArmPosition=70, leftArmPosition=70, leftArmVelocity=speed_hands, rightArmVelocity=speed_hands, units="degrees")
    self.ws.close()
    


  #### PROGRAM:
 
  def open_websocket(self):
    
    self.ws = websocket.WebSocketApp("ws://" + self.ip + "/pubsub", 
                                on_open=self.on_open,
                                on_message=self.on_message,
                                on_error=self.on_error,
                                on_close=self.on_close)
    
    self.ws.run_forever() 


  def execute_behavior(self, motion_dict, scene_name):
    ws_thread = threading.Thread(target=self.open_websocket)
    ws_thread.start()
    
    # choose a mapping here for the different motions - for testing with children
    
    choices = {"dance_robot": ["direct", "direct"], "dance_disco": ["combo", "direct"], "dance_spooky": ["combo", "direct"],
               "dance_stars": ["combo", "direct"], "dance_chicken": ["combo", "direct"], "dance_marcarena": ["combo", "cross"]}
    
    # [method, mapping]
    if "dance" in scene_name:
      method = choices[scene_name][0] # direct/qom/combo
      mapping = choices[scene_name][1] # direct/cross - if it is cross i need to add dir too tho - i think i added it?
    else: 
      method = "direct"
      mapping = "direct"
  
    
    
    pitch_dict, roll_dict, yaw_dict, rarm_dict, larm_dict, wheels, motion_length, song, speed, eyes, leds = self.read_from_yaml(motion_dict, method, mapping, scene_name) 
    
    if mapping == "cross": # find dir
      dir_dict = motion_dict[mapping]["head"]["directions"]
      print("INN HIT")
    else:
      dir_dict = ""
  
    # to test out the different approaches:
    
    if method == "combo":
      self.move_direct(motion_length, pitch_dict, roll_dict, yaw_dict, rarm_dict, larm_dict, wheels, speed, eyes, leds, song, mapping, method, scene_name, dir_dict)
    
    elif method == "direct": 
      self.move_direct(motion_length, pitch_dict, roll_dict, yaw_dict, rarm_dict, larm_dict, wheels, "", eyes, leds, song, mapping, method, scene_name, dir_dict)

    elif method == "qom":
      self.move_intensity(motion_length, pitch_dict, roll_dict, yaw_dict, rarm_dict, larm_dict, wheels, speed, eyes, leds, song, scene_name)
      
    print("kjem hit???")



  def execute_special(self, behavior_name):
    """ Using Misty API to do other tasks on the robot, such as volume, talking etc. """
    if "tts" in behavior_name: # then make misty speak whatever is inside the tts
      phrase = behavior_name.replace("tts ", "")
      language = "Norwegian 4"
      rate = 0.5
      self.misty.speak(text=phrase, language=language, speechRate=rate)
    
    elif behavior_name == "plus": # volume opp 
      vol = 70
      self.misty.set_default_volume(volume=vol) # setting a higher volume
      self.misty.play_audio("notification-37858.mp3", vol)
      
    elif behavior_name == "minus":
      vol = 30
      self.misty.set_default_volume(volume=vol)
      self.misty.play_audio("notification-37858.mp3", vol)
      
    elif behavior_name == "check_connection": # blink led and play sound
      self.misty.play_audio("notification-37858.mp3", volume=50)
      self.misty.change_led(red=0, green=255, blue=0)
      time.sleep(2)
      self.misty.change_led(red=100, green=70, blue=160)
      
    elif behavior_name == "mute":
      self.misty.set_default_volume(volume=0)
      
    elif behavior_name == "unmute":
      self.misty.set_default_volume(volume=50)
      
    else: 
      print(f"Behavior {behavior_name} is not executed.")
      
      

  def execute(self):
    # her inne kan jeg ha liksom tasks depending on the ka melding det e som kommer inn (component name)    
    # i denne metoden sender vi kommandoa til misty - messagen bil delen
    # {component1: {..}, component2.. etc} - {arms, head, wheels}
  
    decoded_message = base64.b64decode(self.payload).decode("utf-8")
    msg_list = decoded_message.split("^") # splitting message to get everything
    
    scene_name = msg_list[0]
    behavior_dictionary = ast.literal_eval(msg_list[2])
    
    if "tag" in behavior_dictionary.keys(): # checking if we have a special behavior
      self.execute_special(behavior_dictionary["behavior"])
    else:
      self.execute_behavior(behavior_dictionary, scene_name)
  
   
    return self.active

  def abort(self):
    self.active = False
    return True



