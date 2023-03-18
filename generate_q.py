import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time
import random
import pdb

human_controller = False

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# Let's add some sidewalks and RectangleBuildings.
# A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
# A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
# For both of these objects, we give the center point and the size.
w.add(Painting(Point(71.5, 106.5), Point(97, 27), 'gray80')) # We build a sidewalk.
w.add(RectangleBuilding(Point(72.5, 107.5), Point(95, 25))) # The RectangleBuilding is then on top of the sidewalk, with some margin.

# Let's repeat this for 4 different RectangleBuildings.
w.add(Painting(Point(8.5, 106.5), Point(17, 27), 'gray80'))
w.add(RectangleBuilding(Point(7.5, 107.5), Point(15, 25)))

w.add(Painting(Point(8.5, 41), Point(17, 82), 'gray80'))
w.add(RectangleBuilding(Point(7.5, 40), Point(15, 80)))

w.add(Painting(Point(71.5, 41), Point(97, 82), 'gray80'))
w.add(RectangleBuilding(Point(72.5, 40), Point(95, 80)))

# Let's also add some zebra crossings, because why not.
w.add(Painting(Point(18, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(19, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(20, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(21, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(22, 81), Point(0.5, 2), 'white'))


# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
lightning_mcqueen = Car(Point(20,20), np.pi/2)
w.add(lightning_mcqueen)

#c2 = Car(Point(118,90), np.pi, 'blue')
mater = Car(Point(20, 30), np.pi/2, 'blue')
mater.velocity = Point(3.0,0) # We can also specify an initial velocity just like this.
w.add(mater)

mykel = Car(Point(20,0), np.pi/2, 'yellow')
mykel.velocity = Point(3.0, 0)
w.add(mykel)

# Pedestrian is almost the same as Car. It is a "circle" object rather than a rectangle.
#p1 = Pedestrian(Point(28,81), np.pi)
#p1.max_speed = 10.0 # We can specify min_speed and max_speed of a Pedestrian (and of a Car). This is 10 m/s, almost Usain Bolt.
#w.add(p1)
cars = [lightning_mcqueen, mater, mykel]
#w.render() # This visualizes the world we just constructed.

#initialize things needed for Q learning
#1. intialize actions
actions = [0, 1, 2] # slow, speed, stay constant 
possible_states = [0, 1, 2, 3] #collision, too close, just right, too far 

Q_table = np.zeros((3, 4, 3))
alpha = 0.001
gamma = 0.2
epsilon = 0.1 

'''Define step function from one state to the next'''
def step(cur_state, action):
    next_state = cur_state
    if action == 0: 
        if cur_state != 3: 
            next_state = cur_state + 1
    if action == 1: 
        if cur_state != 0: 
            next_state = cur_state - 1 
    return next_state

'''Define reward function for each state'''
def cur_reward(new_state):
    reward = 0
    if new_state == 0: # collision
        reward = -100 
    if new_state == 1: # too close
        reward = -35
    if new_state == 2: # just right
        reward = 50
    if new_state == 3: # too far
        reward = -50
        
    return reward 

for i in range(10000): 
    states_of_cars = [2, 2, 2]
    epochs, penalties, reward = 0, 0, 0
    for i in range(500): 
       # next_state = [r for r in step(cur_state, action)]
        for car_index in range(len(cars)):
            cur_state = states_of_cars[car_index]
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions) # Explore action space
            else:
                action = np.argmax(Q_table[car_index][states_of_cars[car_index]]) # Exploit learned values
            new_state = step(cur_state, action)
            reward = cur_reward(new_state)
            old_value = Q_table[car_index][cur_state, action]
            next_max = np.max(Q_table[car_index][new_state])
            new_value =  (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            Q_table[car_index][cur_state, action] = new_value

            if reward == -100: # num collisions 
                penalties += 1

            state = new_state
            states_of_cars[car_index] = state
            epochs += 1

print(Q_table)
print(penalties)
pdb.set_trace()


# if not human_controller:
#     # Let's implement some simple scenario with all agents
#     #p1.set_control(0, 0.22) # The pedestrian will have 0 steering and 0.22 throttle. So it will not change its direction.
#     c1.set_control(0, 0.35)
#     mater.set_control(0, 0.05)
#     mykel.set_control(0, 0.5)
#     print("c1 to c2 is", c1.distanceTo(mater))
#     print("c2 to c3 is", mater.distanceTo(mykel))
#     print("c1 to c3 is", c1.distanceTo(mykel))
#     for k in range(400):
#         # All movable objects will keep their control the same as long as we don't change it.
#         if k == 100: # Let's say the first Car will release throttle (and start slowing down due to friction)
#             c1.set_control(0, 0)
#         elif k == 200: # The first Car starts pushing the brake a little bit. The second Car starts turning right with some throttle.
#             c1.set_control(0, -0.02)
#         elif k == 325:
#             c1.set_control(0, 0.8)
#             mater.set_control(-0.45, 0.3)
#         elif k == 367: # The second Car stops turning.
#             mater.set_control(0, 0.1)
#         w.tick() # This ticks the world for one time step (dt second)
#         w.render()
#         time.sleep(dt/4) # Let's watch it 4x

#         #if w.collision_exists(p1): # We can check if the Pedestrian is currently involved in a collision. We could also check c1 or c2.
#             #print('Pedestrian has died!')
#         if w.collision_exists(): # Or we can check if there is any collision at all.
#             print('Collision exists somewhere...')
#     w.close()

# else: # Let's use the steering wheel (Logitech G29) for the human control of car c1
#     #p1.set_control(0, 0.22) # The pedestrian will have 0 steering and 0.22 throttle. So it will not change its direction.
#     mater.set_control(0, 0.35)
    
#     from interactive_controllers import SteeringWheelController
#     controller = SteeringWheelController(w)
#     for k in range(400):
#         c1.set_control(controller.steering, controller.throttle)
#         w.tick() # This ticks the world for one time step (dt second)
#         w.render()
#         time.sleep(dt/4) # Let's watch it 4x
#         if w.collision_exists():
#             import sys
#             sys.exit(0)