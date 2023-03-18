import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time
import random
import pdb

q_table = np.array([
  [
    [-22.41863358, -29.92341784, -30.81256449],
    [62.5, -104.02446061, -22.39827434],
    [-37.5, -22.5, 62.5],
    [-37.34230315, 62.5, -37.35475905]
  ],
  [
    [-22.40555398, -28.95124489, -29.25422955],
    [62.5, -103.95413466, -22.40726823],
    [-37.5, -22.5, 62.5],
    [-37.34820736, 62.5, -37.34777155]
  ],
  [
    [-22.41894399, -29.40277942, -31.25597933],
    [62.5, -104.02939146, -22.41729132],
    [-37.5, -22.5, 62.5],
    [-37.33577621, 62.5, -37.35785331]
  ]
])

q_dumb_table =  np.random.rand(3, 4, 3)


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


 # This visualizes the world we just constructed.

def min_distance_to_cars(): 
    dist_arr = []
    for car_i in range(len(cars)): 
        car_distances = [cars[car_i].distanceTo(cars[i]) for i in range(len(cars)) if i != car_i]
        dist_arr.append(min(car_distances))
    
    return dist_arr

def bucket_all(min_distances): 
    states = [0] * len(min_distances)
    for i in range(len(min_distances)):
        states[i] = bucket_one(min_distances[i])
    return states 


def bucket_one(min_distance): 
    # in: min_distances
    # out: states for each car
    state = 0
    if min_distance < 1: 
        state = 0
    elif min_distance < 1.5: 
        state = 1
    elif min_distance < 6: 
        state = 2
    else:
        state = 3 
    return state 

states =[2, 2, 2]

def change_vel(optimal_action, car_index): 
    curr_velocity = cars[car_index].velocity.x
    if optimal_action == 0: 
        cars[car_index].velocity = Point(curr_velocity - 1.5, 0)
    if optimal_action == 1: 
        cars[car_index].velocity = Point(curr_velocity + 2.5, 0)

# Q LEARNED POLICY
count_optimal_state = 0
count_collisions = 0
epochs = 20
for j in range(epochs): 
    print("starting epoch {j}")
    random_velocities = np.random.uniform(7., 25., size=3)
    starting_position_y = np.random.uniform(0., 9.)
    mykel = Car(Point(20,starting_position_y), np.pi/2, 'yellow')
    mykel.velocity = Point(random_velocities[0], 0)
    w.add(mykel)

    starting_position_y = np.random.uniform(11, 19)
    # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
    lightning_mcqueen = Car(Point(20,starting_position_y), np.pi/2)
    lightning_mcqueen.velocity = Point(random_velocities[1], 0)
    w.add(lightning_mcqueen)

    starting_position_y = np.random.uniform(21, 30)
    #c2 = Car(Point(118,90), np.pi, 'blue')
    mater = Car(Point(20, starting_position_y), np.pi/2, 'blue')
    mater.velocity = Point(random_velocities[2],0) # We can also specify an initial velocity just like this.
    w.add(mater)
    cars = [mykel, lightning_mcqueen, mater]

    # w.render()
    for k in range(400): 
        cur_count_collisions = 0
        cur_count_optimal_state = 0
        # get min distance to other cars 
        min_distances = min_distance_to_cars()
        # bucket the distance to put car in a state 
        states = bucket_all(min_distances)
        if 0 in states:
            cur_count_collisions += 1
        if (states[0] == 2) and (states[1] == 2) and (states[1] == 2): 
            cur_count_optimal_state += 1
            for car in cars: 
                car.velocity = Point(30, 0)
        # pick best action for car at that state 
        for car_index in range(len(cars)): 
            car_i_q = q_table[car_index]
            optimal_action = np.argmax(car_i_q[states[car_index]])
            change_vel(optimal_action, car_index)
        w.tick()
       # w.render()
        time.sleep(dt/4)
      #  print(states)
    print("ending epoch {j}")
    count_collisions += cur_count_collisions
    count_optimal_state += cur_count_optimal_state

f = open("results_real.txt", "w")
f.write("RESULTS FROM REAL QLEARNING POLICY\n") 
f.write(f"collision rate: {count_collisions/epochs}, optimal state rate: {count_optimal_state/epochs}")
f.close()

# DUMMY BASELIEN OF RANDOM POLICY
count_optimal_state = 0
count_collisions = 0
epochs = 20
for j in range(epochs): 
    print("starting epoch {j}")
    random_velocities = np.random.uniform(7., 25., size=3)
    starting_position_y = np.random.uniform(0., 9.)
    mykel = Car(Point(20,starting_position_y), np.pi/2, 'yellow')
    mykel.velocity = Point(random_velocities[0], 0)
    w.add(mykel)

    starting_position_y = np.random.uniform(11, 19)
    # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
    lightning_mcqueen = Car(Point(20,starting_position_y), np.pi/2)
    lightning_mcqueen.velocity = Point(random_velocities[1], 0)
    w.add(lightning_mcqueen)

    starting_position_y = np.random.uniform(21, 30)
    #c2 = Car(Point(118,90), np.pi, 'blue')
    mater = Car(Point(20, starting_position_y), np.pi/2, 'blue')
    mater.velocity = Point(random_velocities[2],0) # We can also specify an initial velocity just like this.
    w.add(mater)
    cars = [mykel, lightning_mcqueen, mater]

#    w.render()
    for k in range(400): 
        cur_count_collisions = 0
        cur_count_optimal_state = 0
        # get min distance to other cars 
        min_distances = min_distance_to_cars()
        # bucket the distance to put car in a state 
        states = bucket_all(min_distances)
        if 0 in states:
            cur_count_collisions += 1
        if (states[0] == 2) and (states[1] == 2) and (states[1] == 2): 
            cur_count_optimal_state += 1
            for car in cars: 
                car.velocity = Point(30, 0)
        # pick best action for car at that state 
        for car_index in range(len(cars)): 
            car_i_q = q_dumb_table[car_index]
            optimal_action = np.argmax(car_i_q[states[car_index]])
            change_vel(optimal_action, car_index)
        w.tick()
        #w.render()
        time.sleep(dt/4)
       # print(states)
    print("ending epoch {j}")
    count_collisions += cur_count_collisions
    count_optimal_state += cur_count_optimal_state

f = open("results_dumb.txt", "w")
f.write("RESULTS FROM REAL QLEARNING POLICY\n") 
f.write(f"collision rate: {count_collisions/4000}, optimal state rate: {count_optimal_state/4000}")
f.close()