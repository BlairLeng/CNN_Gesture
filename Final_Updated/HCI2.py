import pyautogui as pag
def HCI(r):
    print("here")
    screen = pag.size()
    c = pag.position()
    seconds = 1

    clickflag = True
    dragflag = True
    #info = [0, 1, 0, 0, 0, 0, 0, 1, 0]

    #fist = start
    if r == 1:
        flag = True

    #if flag:
        #info = [failure, fist, up, down, left, right, none, 2, 3]
            
        #peace = click
    if r == [7] and clickflag:
        print("click")
        pag.click(button = "left")
        clickFlag = False

        #gesture three = click twice
    elif r == [1] and dragflag:
        print("double click")
        pag.click(clicks = 2)
        clickFlag = True
            
        #up
    elif r == [2]:
        print("up")
        pag.moveTo(c[0], c[1] - 10)

        #down
    elif r == [3]:
        print("down")
        pag.moveTo(c[0], c[1] + 10)

        #left
    elif r == [4]:
        print("left")
        pag.moveTo(c[0] - 10, c[1])

        #right
    elif r == [5]:
        print("right")
        pag.moveTo(c[0] + 10, c[1])

