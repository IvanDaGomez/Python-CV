import time
# function to simulate an attendance system
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = time.time()
            date = time.ctime(now)
            f.writelines(f'\n{name},{date}')