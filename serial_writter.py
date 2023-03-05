import serial, time

arduino = serial.Serial('COM9', 115200, timeout=.1)

time.sleep(0.5)

arduino.write(b"aqq")
time.sleep(0.2)
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
        elif sig_==-1:
            sigstr += "-"
    sigstr += "1"
    arduino.write(sigstr.encode('utf-8'))
 

signal((0, 0, 0, 0))