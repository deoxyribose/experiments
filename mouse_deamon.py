from pymouse import PyMouse
from pykeyboard import PyKeyboard
import numpy as np
import daemon
import time
import sys

#k = PyKeyboard()

#def record_mouse_on_movement(how_many_seconds = 3): 
#    mouse_positions = [(0,0)]
#    t = []
#    end = time.time() + how_many_seconds
#    current_time = 0
#    while current_time < end:
#        current_time = time.time()
#        current_mouse_position = m.position()
#        if current_mouse_position != mouse_positions[-1]:
#            t.append(current_time)
#            mouse_positions.append(current_mouse_position)
#    return t, np.array(mouse_positions[1:])

def recreate_mouse_movement(timestamps, mouse_positions):
    assert len(timestamps) == mouse_positions.shape[0]
    time_differences = np.diff(timestamps)
    for t,mouse_position in zip(time_differences,mouse_positions):
        #print(t,mouse_position)
        time.sleep(t)
        m.move(*mouse_position)

def flip_upside_down(mouse_positions):
    xmax, ymax = m.screen_size()
    return mouse_positions*np.array([1,-1]) + np.array([0,ymax])

def record_mouse_on_movement():
    m = PyMouse()
    last_mouse_position = (0,0)
    while True:
        current_time = time.time()
        current_mouse_position = m.position()
        if current_mouse_position != last_mouse_position:
            with open('/tmp/mouse.csv', 'a') as fh:
                fh.write("\n")
                fh.write(str(current_time))
                fh.write(" ")
                fh.write(str(current_mouse_position[0]))
                fh.write(" ")
                fh.write(str(current_mouse_position[1]))
                fh.write(" ")
            last_mouse_position = current_mouse_position

with daemon.DaemonContext(
        stdout=sys.stdout,
        stderr=sys.stderr):
    record_mouse_on_movement()