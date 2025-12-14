from rubiks import Rubik

def main():
    my_cube = Rubik()

    SCRAMBLE_MOVES = 3
    while True:
        try:
            SCRAMBLE_MOVES = input("Please enter number of scramble moves: ")
            SCRAMBLE_MOVES = int(SCRAMBLE_MOVES)
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    while True:
        try:
            DEPTH = input("Please enter depth: ")
            DEPTH = int(DEPTH)
            if DEPTH > 7:
                print("Input is too large.")
                continue
            break
        except ValueError:
            print("Invalid input.")
        
    my_cube.scramble(num_moves=SCRAMBLE_MOVES)

    print("\n--- State After Scramble ---")
    my_cube.display()

    solution = my_cube.solve(max_moves=DEPTH)

    if solution:
        my_cube.apply_moves(solution)
        print("\n--- State After Applying Solution ---")
        my_cube.display()
        print(f"Is cube solved? {my_cube.is_solved()}")
    else:
        print("No solution found.")


if __name__ == "__main__":
    main() 