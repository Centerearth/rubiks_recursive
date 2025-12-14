import copy
import random # <-- NEW IMPORT

class Rubik:
    """
    Represents the state of a 3x3 Rubik's Cube.
    ... existing docstring ...
    """
    
    def __init__(self, initial_state=None):
        """
        Initializes the cube.
        If 'initial_state' is provided, it sets the cube to that state.
        If not, it initializes to a solved state by calling
        the internal _get_solved_state_dict() method.
        """
        if initial_state:
            # Use deepcopy to ensure the provided state is not
            # modified by subsequent moves in this instance.
            self.cube = copy.deepcopy(initial_state)
        else:
            # Default to a solved state if none is provided
            self.cube = self._get_solved_state_dict()

        # --- NEW ---
        # Map of "nice" move names to the actual method names.
        # This is essential for the recursive solver.
        self.MOVE_MAPPING = {
            'F': 'move_f_clockwise',
            'F_inv': 'move_f_counter_clockwise',
            'U': 'move_u_clockwise',
            'U_inv': 'move_u_counter_clockwise',
            'D': 'move_d_clockwise',
            'D_inv': 'move_d_counter_clockwise',
            'R': 'move_r_clockwise',
            'R_inv': 'move_r_counter_clockwise',
            'L': 'move_l_clockwise',
            'L_inv': 'move_l_counter_clockwise',
            'B': 'move_b_clockwise',
            'B_inv': 'move_b_counter_clockwise'
        }

    def _get_solved_state_dict(self):
        """
        Helper method to generate and return a dictionary 
        representing a solved cube in the standard WCA color scheme.
        """
        # Define the colors for each face in a solved state
        # W = White, Y = Yellow, G = Green, B = Blue, R = Red, O = Orange
        
        solved_up =    [['W', 'W', 'W'],
                        ['W', 'W', 'W'],
                        ['W', 'W', 'W']]
                        
        solved_down =  [['Y', 'Y', 'Y'],
                        ['Y', 'Y', 'Y'],
                        ['Y', 'Y', 'Y']]
                        
        solved_front = [['G', 'G', 'G'],
                        ['G', 'G', 'G'],
                        ['G', 'G', 'G']]
                        
        solved_back =  [['B', 'B', 'B'],
                        ['B', 'B', 'B'],
                        ['B', 'B', 'B']]
                        
        solved_right = [['R', 'R', 'R'],
                        ['R', 'R', 'R'],
                        ['R', 'R', 'R']]
                        
        solved_left =  [['O', 'O', 'O'],
                        ['O', 'O', 'O'],
                        ['O', 'O', 'O']]

        # Return the complete solved state dictionary
        return {
            'U': solved_up,
            'D': solved_down,
            'F': solved_front,
            'B': solved_back,
            'R': solved_right,
            'L': solved_left
        }

    # --- NEW METHOD ---
    def is_solved(self):
        """
        Checks if the cube is in a solved state.
        A cube is solved if all stickers on a face match the center
        sticker of that face, for all 6 faces.
        """
        try:
            for face_name in self.cube:
                # Get the center sticker's color (at [1][1])
                # This is the "target" color for the face.
                center_color = self.cube[face_name][1][1]
                
                # Check all 9 stickers on the face
                for r in range(3):
                    for c in range(3):
                        if self.cube[face_name][r][c] != center_color:
                            return False  # Mismatch found
        except (KeyError, IndexError, TypeError):
            # This handles cases of a malformed or incomplete cube state
            print("Error: Cube state is malformed.")
            return False
            
        return True # No mismatches found on any face

    # --- REVERTED FUNCTION ---
    def display(self):
        """
        Prints a simple 2D text representation of the cube's state.
        This is very helpful for debugging.
        """
        print("--- Cube State ---")
        
        # Display Up face
        print("      Up Face (U)")
        for row in self.cube['U']:
            print("      " + " ".join(row))
        print("-" * 20)
        
        # Display Left, Front, Right, and Back faces in a row
        print("Left (L)  Front (F) Right (R) Back (B)")
        for i in range(3):
            # Get the i-th row from each face
            l_row = " ".join(self.cube['L'][i])
            f_row = " ".join(self.cube['F'][i])
            r_row = " ".join(self.cube['R'][i])
            b_row = " ".join(self.cube['B'][i])
            
            print(f"{l_row} | {f_row} | {r_row} | {b_row}")
        print("-" * 20)

        # Display Down face
        print("      Down Face (D)")
        for row in self.cube['D']:
            print("      " + " ".join(row))
        print("------------------\n")
        
    def get_state(self):
        """
        Returns a deep copy of the current cube state.
        This is crucial for recursion, so you can pass a
        new state without modifying the old one.
        """
        return copy.deepcopy(self.cube)

    def set_state(self, new_state):
        """
        Sets the cube's state to a new state.
        """
        self.cube = new_state

    # --- NEW HELPER METHOD ---
    def _rotate_face_clockwise(self, face_name):
        """
        Rotates the 9 stickers on a single given face clockwise.
        """
        face = self.cube[face_name]
        new_face = [['', '', ''], ['', '', ''], ['', '', '']]
        
        # New corners
        new_face[0][0] = face[2][0]
        new_face[0][2] = face[0][0]
        new_face[2][2] = face[0][2]
        new_face[2][0] = face[2][2]
        
        # New edges
        new_face[0][1] = face[1][0]
        new_face[1][2] = face[0][1]
        new_face[2][1] = face[1][2]
        new_face[1][0] = face[2][1]
        
        # Center
        new_face[1][1] = face[1][1]
        
        self.cube[face_name] = new_face

    # --- CLOCKWISE MOVES ---
    
    def move_f_clockwise(self):
        """
        Performs a clockwise rotation of the Front (F) face.
        This modifies the object's internal state.
        """
        
        # --- Part 1: Rotate the Front face stickers ---
        self._rotate_face_clockwise('F')
        
        # --- Part 2: Move the adjacent stickers ---
        # The cycle is: U -> R -> D -> L -> U
        
        # Store the bottom row of 'U' temporarily, as it will be overwritten.
        # We use [:] to create a shallow copy.
        temp_up_row = self.cube['U'][2][:]
        
        # 1. Up (bottom row) <- Left (right column, reversed)
        self.cube['U'][2][0] = self.cube['L'][2][2]
        self.cube['U'][2][1] = self.cube['L'][1][2]
        self.cube['U'][2][2] = self.cube['L'][0][2]
        
        # 2. Left (right column) <- Down (top row)
        self.cube['L'][0][2] = self.cube['D'][0][0]
        self.cube['L'][1][2] = self.cube['D'][0][1]
        self.cube['L'][2][2] = self.cube['D'][0][2]
        
        # 3. Down (top row) <- Right (left column, reversed)
        self.cube['D'][0][0] = self.cube['R'][2][0]
        self.cube['D'][0][1] = self.cube['R'][1][0]
        self.cube['D'][0][2] = self.cube['R'][0][0]
        
        # 4. Right (left column) <- Temp (original Up bottom row)
        self.cube['R'][0][0] = temp_up_row[0]
        self.cube['R'][1][0] = temp_up_row[1]
        self.cube['R'][2][0] = temp_up_row[2]

    def move_u_clockwise(self):
        """ Performs a clockwise rotation of the Up (U) face. """
        
        # Part 1: Rotate the face itself
        self._rotate_face_clockwise('U')
        
        # Part 2: Move adjacent stickers
        # Cycle: F -> R -> B -> L -> F
        temp_row = self.cube['F'][0][:]
        self.cube['F'][0] = self.cube['L'][0][:]
        self.cube['L'][0] = self.cube['B'][0][:]
        self.cube['B'][0] = self.cube['R'][0][:]
        self.cube['R'][0] = temp_row

    def move_d_clockwise(self):
        """ Performs a clockwise rotation of the Down (D) face. """
        
        # Part 1: Rotate the face itself
        self._rotate_face_clockwise('D')
        
        # Part 2: Move adjacent stickers
        # Cycle: F -> L -> B -> R -> F
        temp_row = self.cube['F'][2][:]
        self.cube['F'][2] = self.cube['R'][2][:]
        self.cube['R'][2] = self.cube['B'][2][:]
        self.cube['B'][2] = self.cube['L'][2][:]
        self.cube['L'][2] = temp_row

    def move_r_clockwise(self):
        """ Performs a clockwise rotation of the Right (R) face. """
        
        # Part 1: Rotate the face itself
        self._rotate_face_clockwise('R')
        
        # Part 2: Move adjacent stickers
        # Cycle: U -> B -> D -> F -> U
        # Note: B and U faces are indexed differently (reversed)
        temp_col = [self.cube['U'][0][2], self.cube['U'][1][2], self.cube['U'][2][2]]
        
        # U (right col) <- F (right col)
        self.cube['U'][0][2] = self.cube['F'][0][2]
        self.cube['U'][1][2] = self.cube['F'][1][2]
        self.cube['U'][2][2] = self.cube['F'][2][2]
        
        # F (right col) <- D (right col)
        self.cube['F'][0][2] = self.cube['D'][0][2]
        self.cube['F'][1][2] = self.cube['D'][1][2]
        self.cube['F'][2][2] = self.cube['D'][2][2]

        # D (right col) <- B (left col, reversed)
        self.cube['D'][0][2] = self.cube['B'][2][0]
        self.cube['D'][1][2] = self.cube['B'][1][0]
        self.cube['D'][2][2] = self.cube['B'][0][0]
        
        # B (left col) <- Temp (U right col, reversed)
        self.cube['B'][0][0] = temp_col[2]
        self.cube['B'][1][0] = temp_col[1]
        self.cube['B'][2][0] = temp_col[0]

    def move_l_clockwise(self):
        """ Performs a clockwise rotation of the Left (L) face. """
        
        # Part 1: Rotate the face itself
        self._rotate_face_clockwise('L')
        
        # Part 2: Move adjacent stickers
        # Cycle: U -> F -> D -> B -> U
        # Note: B and D faces are indexed differently (reversed)
        temp_col = [self.cube['U'][0][0], self.cube['U'][1][0], self.cube['U'][2][0]]
        
        # U (left col) <- B (right col, reversed)
        self.cube['U'][0][0] = self.cube['B'][2][2]
        self.cube['U'][1][0] = self.cube['B'][1][2]
        self.cube['U'][2][0] = self.cube['B'][0][2]

        # B (right col) <- D (left col, reversed)
        self.cube['B'][0][2] = self.cube['D'][2][0]
        self.cube['B'][1][2] = self.cube['D'][1][0]
        self.cube['B'][2][2] = self.cube['D'][0][0]

        # D (left col) <- F (left col)
        self.cube['D'][0][0] = self.cube['F'][0][0]
        self.cube['D'][1][0] = self.cube['F'][1][0]
        self.cube['D'][2][0] = self.cube['F'][2][0]

        # F (left col) <- Temp (U left col)
        self.cube['F'][0][0] = temp_col[0]
        self.cube['F'][1][0] = temp_col[1]
        self.cube['F'][2][0] = temp_col[2]
        
    def move_b_clockwise(self):
        """ Performs a clockwise rotation of the Back (B) face. """
        
        # Part 1: Rotate the face itself
        self._rotate_face_clockwise('B')
        
        # Part 2: Move adjacent stickers
        # Cycle: U -> L -> D -> R -> U
        # Note: All adjacent faces are indexed differently (transposed/reversed)
        temp_row = self.cube['U'][0][:]

        # U (top row) <- R (right col, reversed)
        self.cube['U'][0][0] = self.cube['R'][2][2]
        self.cube['U'][0][1] = self.cube['R'][1][2]
        self.cube['U'][0][2] = self.cube['R'][0][2]

        # R (right col) <- D (bottom row)
        self.cube['R'][0][2] = self.cube['D'][2][0]
        self.cube['R'][1][2] = self.cube['D'][2][1]
        self.cube['R'][2][2] = self.cube['D'][2][2]

        # D (bottom row) <- L (left col, reversed)
        self.cube['D'][2][0] = self.cube['L'][2][0]
        self.cube['D'][2][1] = self.cube['L'][1][0]
        self.cube['D'][2][2] = self.cube['L'][0][0]

        # L (left col) <- Temp (U top row)
        self.cube['L'][0][0] = temp_row[0]
        self.cube['L'][1][0] = temp_row[1]
        self.cube['L'][2][0] = temp_row[2]

    # --- COUNTER-CLOCKWISE (INVERSE) MOVES ---
    
    def move_f_counter_clockwise(self):
        """ Performs a counter-clockwise rotation of the Front (F) face. """
        self.move_f_clockwise()
        self.move_f_clockwise()
        self.move_f_clockwise()

    def move_u_counter_clockwise(self):
        """ Performs a counter-clockwise rotation of the Up (U) face. """
        self.move_u_clockwise()
        self.move_u_clockwise()
        self.move_u_clockwise()

    def move_d_counter_clockwise(self):
        """ Performs a counter-clockwise rotation of the Down (D) face. """
        self.move_d_clockwise()
        self.move_d_clockwise()
        self.move_d_clockwise()

    def move_r_counter_clockwise(self):
        """ Performs a counter-clockwise rotation of the Right (R) face. """
        self.move_r_clockwise()
        self.move_r_clockwise()
        self.move_r_clockwise()

    def move_l_counter_clockwise(self):
        """ Performs a counter-clockwise rotation of the Left (L) face. """
        self.move_l_clockwise()
        self.move_l_clockwise()
        self.move_l_clockwise()

    def move_b_counter_clockwise(self):
        """ Performs a counter-clockwise rotation of the Back (B) face. """
        self.move_b_clockwise()
        self.move_b_clockwise()
        self.move_b_clockwise()

    # --- NEW SOLVER & HELPER METHODS ---
    
    def _get_inverse_move_name(self, move_name):
        """Helper to get the inverse move name."""
        if move_name.endswith('_inv'):
            return move_name[:-4]
        return move_name + '_inv'

    def scramble(self, num_moves=20):
        """
        Applies a number of random moves to scramble the cube.
        """
        move_names = list(self.MOVE_MAPPING.keys())
        last_face = '' # To avoid redundant moves (e.g., F B F')
        scramble_moves = []
        
        for _ in range(num_moves):
            while True:
                move_name = random.choice(move_names)
                current_face = move_name[0] # e.g., 'F' from 'F' or 'F_inv'
                
                # Simple pruning: Don't move the same face twice in a row
                if current_face != last_face:
                    last_face = current_face
                    scramble_moves.append(move_name)
                    break
        
        print(f"Scrambling with {num_moves} moves: {' '.join(scramble_moves)}")
        self.apply_moves(scramble_moves)

    def apply_moves(self, move_list):
        """
        Applies a list of move names (e.g., ['F', 'U_inv'])
        to the current cube state.
        """
        for move_name in move_list:
            if move_name in self.MOVE_MAPPING:
                method_name = self.MOVE_MAPPING[move_name]
                move_function = getattr(self, method_name)
                move_function()
            else:
                print(f"Warning: Unknown move '{move_name}' skipped.")

    def solve(self, max_moves=7):
        """
        Attempts to solve the cube using Iterative Deepening
        Depth-First Search (IDDFS).
        
        Starts searching for a solution of depth 1, then 2, etc.,
        up to 'max_moves'. This guarantees the first solution
        found is the shortest.
        
        NOTE: This simple algorithm is EXTREMELY slow and will
        only find solutions for shallow scrambles (e.g., < 7 moves).
        A full "God's Number" solver is vastly more complex.
        """
        print(f"Attempting to solve in a maximum of {max_moves} moves...")
        
        # Save the initial state so we can restore it after the search.
        # This allows the recursive search to modify the cube in-place (much faster).
        initial_state_backup = self.get_state()

        # Iterative Deepening: Search at depth 0, 1, 2, ...
        for depth in range(max_moves + 1):
            print(f"  Searching at depth: {depth}")
            
            # Perform search (modifies self in-place, then backtracks)
            solution_path = self._recursive_search([], depth)
            
            if solution_path is not None:
                print(f"\nSolution found in {len(solution_path)} moves!")
                print(f"  Path: {' '.join(solution_path)}")
                # Restore the cube to its original state before returning
                self.set_state(initial_state_backup)
                return solution_path
                
        # Restore state if no solution found
        self.set_state(initial_state_backup)
        print(f"\nNo solution found within {max_moves} moves.")
        return None

    def _recursive_search(self, path, depth_limit):
        """
        Recursive helper for the 'solve' method.
        Uses backtracking to avoid expensive deep copies.
        """
        
        # --- Base Case 1: Success ---
        if self.is_solved():
            return path # We found a solution!
            
        # --- Base Case 2: Failure (Depth Limit Reached) ---
        if len(path) == depth_limit:
            return None # Reached limit, backtrack
            
        # --- Pruning Setup ---
        # We need to prune redundant moves
        last_move = path[-1] if path else None
        
        # --- Recursive Step: Try all moves ---
        for move_name, method_name in self.MOVE_MAPPING.items():
            
            # --- Pruning Logic ---
            if last_move:
                # 1. Don't undo the last move (e.g., F then F_inv)
                if '_inv' in last_move and last_move.replace('_inv', '') == move_name:
                    continue
                if '_inv' not in last_move and last_move + '_inv' == move_name:
                    continue
                
                # 2. Don't do 3x of the same move (e.g., F F F is F_inv)
                # This also stops F F F F (which is useless)
                if (len(path) >= 2 and 
                    path[-1] == last_move and 
                    path[-2] == last_move and 
                    move_name == last_move):
                    continue
                
                # 3. Commutativity Pruning
                # Avoid searching redundant paths like U D vs D U.
                # We enforce a specific order for opposite faces.
                current_face = move_name[0]
                last_face = last_move[0]
                
                # Block: F after B, R after L, U after D
                if (last_face == 'B' and current_face == 'F') or \
                   (last_face == 'L' and current_face == 'R') or \
                   (last_face == 'D' and current_face == 'U'):
                    continue

            # --- Apply Move (In-Place) ---
            move_function = getattr(self, method_name)
            move_function()
            
            # --- Recurse ---
            result = self._recursive_search(path + [move_name], depth_limit)
            
            if result is not None:
                return result # Pass the solution up the chain
            
            # --- Backtrack: Undo the move ---
            inverse_move = self._get_inverse_move_name(move_name)
            inverse_method = self.MOVE_MAPPING[inverse_move]
            getattr(self, inverse_method)()
        
        # If we loop through all moves and find no solution, backtrack
        return None


# --- Example of how to use the class ---

# 1. Create a new, solved cube
my_cube = Rubik()
print("--- Initial Solved State ---")
my_cube.display()

# 2. Scramble it with a *small* number of moves
# (A large scramble is impossible for this simple solver)
SCRAMBLE_MOVES = 3
my_cube.scramble(num_moves=SCRAMBLE_MOVES)
print("\n--- State After Scramble ---")
my_cube.display()
print(f"Is cube solved? {my_cube.is_solved()}") # Should be False

# 3. Try to solve it
# We set max_moves equal to the scramble for a guaranteed quick find
solution = my_cube.solve(max_moves=SCRAMBLE_MOVES)

# 4. If a solution is found, apply it
if solution:
    my_cube.apply_moves(solution)
    print("\n--- State After Applying Solution ---")
    my_cube.display()
    print(f"Is cube solved? {my_cube.is_solved()}") # Should be True!

# 5. Example of a failed solve
print("\n--- Testing a Failed Solve ---")
my_cube.scramble(num_moves=7) # Scramble with 5
solution_fail = my_cube.solve(max_moves=7) # Try to solve in 5
print(f"Solution found: {solution_fail}") # Will be None
