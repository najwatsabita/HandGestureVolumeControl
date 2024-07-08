# HandGestureVolumeControl
This project allows you to control your laptop's volume using hand gestures with your index finger and thumb, utilizing MediaPipe, OpenCV, and Pycaw packages. By detecting the distance between the index finger (landmark 8) and thumb (landmark 4), the script maps this distance to the laptop's volume range of -63.5 (0%) to 0 (100%). The volume is adjusted in real-time, where a minimum distance sets the volume to 0 and a maximum distance sets it to 100%, providing an intuitive way to control audio levels with simple hand movements.
