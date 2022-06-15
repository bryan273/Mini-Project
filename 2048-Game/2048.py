from itertools import product
from random import randint
import copy

newmat=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

def start_game():
    global mat
    mat=[]
    for x in range(4):
        mat.append([0]*4)
    print(
    """Desc : 
    a for left
    w for up
    s for down
    d for right
    """)

def check():
    for i in mat:
        if 2048 in i :
            print("WIN")
            quit()
    for i,j in product(range(4),range(4)):
        if mat[i][j]==0:
            break
    else:
        for i,j  in product(range(4),range(3)):
            if mat[i][j]== mat[i][j+1]:
                break
            elif mat[j][i]==[j+1][i]:
                break
            else:
                print("GAME OVER")
                quit()

def re_verse():
    for x in range(4):
        mat[x].reverse()

def transpose_left():
    global mat
    for i , j in product(range(4),range(4)):
        newmat[i][j]=mat[j][3-i]
    mat=copy.deepcopy(newmat)
    return mat

def transpose_right():
    global mat
    for i,j in product(range(4),range(3,-1,-1)):
        newmat[i][3-j]=mat[j][i]
    mat=copy.deepcopy(newmat)
    return mat

def add():
    x=0
    while x<10:
        i,j = randint(0,3),randint(0,3)
        if mat[i][j]==0:
            mat[i][j]=2
            break
        else:
            x+=1
    else:
        for i,j in product(range(4),range(4)):
            if mat[i][j]==0:
                mat[i][j]=2
                break

def right():
    for i in range(4):
        mat[i].sort(key=lambda x:x<1, reverse = True)
        for j in range(3,0,-1):
            x=mat[i][j]
            y=mat[i][j-1]
            if x==y and x>0 and y>0:
                mat[i][j]=x*2
                mat[i][j-1]=0
                mat[i].sort(key=lambda x : x<1 , reverse=True)
            else:
                continue
    return mat

def left():
    re_verse()
    right()
    re_verse()

def up():
    transpose_right()
    right()
    transpose_left()

def down():
    transpose_left()
    right()
    transpose_right()

def printit():
    for x in range(4):
        print()
        for y in range(4):
            print(str(mat[x][y]).rjust(5, ' '),end=' ')
    print()

start_game()
add()
add()
printit()

while True:
    check()
    print()
    n=input('Select Move : ')
    if n=='a':
        left()
        add()
        printit()
        
        continue
    elif n=='d':
        right()
        add()
        printit()
        
        continue
    elif n=='w':
        up()
        add()
        printit()
        
        continue
    elif n=='s':
        down()
        add()
        printit()
        
        continue
    else:
        print("INVALID MOVE")
        continue
    





