DEFAULT_DISCRETE_ACTIONS = [
    [-1, 0, 0, 0],      # 0 left
    [1, 0, 0, 0],       # 1 right
    [0, -1, 0, 0],      # 2 down
    [0, 1, 0, 0],       # 3 up
    [0, 0, -1, 0],      # 4 clockwise
    [0, 0, 1, 0],       # 5 counter-clockwise
    [0, 0, 0, 1]        # 6 shoot
]

MORE_ACTIONS = [
    [0, 0, 0, 0],       # 0 stand

    [-1, -1, -1, 0],    # 1 move diagonally left-down & rotate clockwise
    [-1, -1, 0, 0],     # 2 move diagonally left-down
    [-1, -1, 1, 0],     # 3 move diagonally left-down & rotate counter-clockwise
    [-1, 0, -1, 0],     # 4 move left & rotate clockwise
    [-1, 0, 0, 0],      # 5 move left
    [-1, 0, 1, 0],      # 6 move left & rotate counter-clockwise
    [0, 1, -1, 0],      # 7 move up & rotate clockwise
    [0, 1, 0, 0],       # 8 move up
    [0, 1, 1, 0],       # 9 move up & rotate counter-clockwise
    [0, -1, -1, 0],     # 10 move down & rotate clockwise
    [0, -1, 0, 0],      # 11 move down
    [0, -1, 1, 0],      # 12 move down & rotate counter-clockwise
    [1, 0, -1, 0],      # 13 move right & rotate clockwise
    [1, 0, 0, 0],       # 14 move right
    [1, 0, 1, 0],       # 15 move right & rotate counter-clockwise
    [1, 1, -1, 0],      # 16 move diagonally right-up & rotate clockwise
    [1, 1, 0, 0],       # 17 move diagonally right-up
    [1, 1, 1, 0],       # 18 move diagonally right-up & rotate counter-clockwise

    [0, 0, 0, 1]        # 19 shoot
]