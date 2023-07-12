from djitellopy import tello
from time import sleep
import cv2
import keyboard
import threading as th


drone = tello.Tello()
drone.connect()
print(drone.get_battery())
drone.streamon()


speed = 50


def getSpeed() :
    global speed
    if keyboard.is_pressed("left"): speed -= 10
    if keyboard.is_pressed("right"): speed += 10
    if speed < 0: speed = 0
    return speed


def getInput() :
    left_right, front_back, up_down, clock_counter = 0,0,0,0

    if keyboard.is_pressed("j"): left_right = -speed
    if keyboard.is_pressed("l"): left_right = speed

    if keyboard.is_pressed("i"): front_back = speed
    if keyboard.is_pressed("k"): front_back = -speed

    if keyboard.is_pressed("w"): up_down = speed
    if keyboard.is_pressed("s"): up_down = -speed

    if keyboard.is_pressed("a"): clock_counter = -speed
    if keyboard.is_pressed("d"): clock_counter = speed   
    
    if keyboard.is_pressed("up"): drone.takeoff()
    if keyboard.is_pressed("down"): drone.land()

    return [left_right, front_back, up_down, clock_counter]



def video_stream() :
    while True:
        image = drone.get_frame_read().frame
        image = cv2.resize(image, (640,480))
        cv2.imshow("Image", image)
        cv2.waitKey(1)


def controller():
    while True:
        getSpeed()
        results = getInput()
        drone.send_rc_control(results[0],results[1],results[2],results[3])
        sleep(0.1)
            
        


video_thread = th.Thread(target=video_stream)
controll_thread = th.Thread(target=controller)


if __name__ == '__main__':
    controll_thread.start()
    video_thread.start()
    
