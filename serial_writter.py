import serial, time

arduino = serial.Serial('COM9', 115200, timeout=.1)

time.sleep(1)

arduino.write(b"aqq")
time.sleep(0.5)
# while True:
#     
#     print(arduino.write('hello'.encode('utf-8')))
#     data = arduino.readline()
#     print(data)



def signal(sig):
    sigstr = ""
    sigstr += "0"
    for sig_ in sig:
        if sig_ == 1:
            sigstr += "+"
        elif sig_ == 0:
            sigstr += "."
        else:
            sigstr += "-"
    sigstr += "1"
    sigstr += "   "
    arduino.write(sigstr.encode('utf-8'))
 
for i in range(20):   
    signal((1, -1))
    for i in range(5):
        print(arduino.readline())
    time.sleep(0.1)
    signal((-1, -1))
    for i in range(5):
        print(arduino.readline())
    time.sleep(0.1)
    signal((1, 1))
    for i in range(5):
        print(arduino.readline())
    time.sleep(0.1)

signal((0,0))


