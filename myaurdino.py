import time
import serial
import sys
# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port=sys.argv[1],
    baudrate=9600,
    parity=serial.PARITY_ODD,
    stopbits=serial.STOPBITS_TWO,
    bytesize=serial.SEVENBITS
)

ser.isOpen()

print 'Enter your commands below.\r\nInsert "exit" to leave the application.'

input=1
while 1 :
    print "Here"
    # get keyboard input
    input = raw_input(">> ")
        # Python 3 users
        # input = input(">> ")
    if input == 'exit':
        ser.close()
        exit()
    else:
        # send the character to the device
        # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
        print input
	ser.write(input)
        out = ''
        # let's wait one second before reading output (let's give device time to answer)
        time.sleep(4)

