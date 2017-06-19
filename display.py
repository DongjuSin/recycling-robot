import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import math
import os

gridSize = 10

ground = 1
plot = plt.figure(figsize=(10,10))
axis = plot.add_subplot(111, aspect='equal')
axis.set_xlim([0, 10])
axis.set_ylim([0, 10])

def drawState(trash_x,trash_y,robot_x,robot_y):
    global gridSize
    #trash_x = trash_column
    #trash_y = gridSize - trash_row + 1
    #statusTitle = "Wins: " + str(winCount) + " Losses: " + str(loseCount) + " TotalGame: " + str(numberOfGames)
    #axis.set_title(statusTitle, fontsize=30)
    for p in [
        patches.Rectangle(
            (0,0), 10, 10, facecolor="#000000"
        ),
        patches.Rectangle(
            (trash_x, trash_y), 1, 1, facecolor="red"
        ),
        patches.Rectangle(
            (robot_x, robot_y), 1, 1, facecolor="green"
        ),
    ]:
        axis.add_patch(p)
    plt.show()
drawState(9,9,1,1)
