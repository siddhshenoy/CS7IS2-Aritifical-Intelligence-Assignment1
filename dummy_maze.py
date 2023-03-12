from pyamaze import maze,COLOR

if __name__ == "__main__":
    print("Starting..")
    maze = maze(5, 5)
    maze.CreateMaze(1,5,loopPercent=100,theme=COLOR.dark)
    maze.run()