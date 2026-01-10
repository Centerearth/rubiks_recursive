import copy
import random

class Rubik:
    """
    Represents the state of a 3x3 Rubik's Cube.
    """
    
    def __init__(self, initial_state=None):
        """
        Initializes the cube.
        If 'initial_state' is provided, it sets the cube to that state.
        If not, it initializes to a solved state by calling
        the internal _get_solved_state_dict() method.
        """
        if initial_state:
            self.cube = copy.deepcopy(initial_state)
        else:
            self.cube = self._get_solved_state_dict()
        
        self.all_hashes = {}

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
        W = White, Y = Yellow, G = Green, B = Blue, R = Red, O = Orange
        """
        
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

        return {
            'U': solved_up,
            'D': solved_down,
            'F': solved_front,
            'B': solved_back,
            'R': solved_right,
            'L': solved_left
        }

    def is_solved(self):
        """
        Checks if the cube is in a solved state.
        A cube is solved if all stickers on a face match the center
        sticker of that face, for all 6 faces.
        """
        try:
            for face_name in self.cube:
                center_color = self.cube[face_name][1][1]
                for r in range(3):
                    for c in range(3):
                        if self.cube[face_name][r][c] != center_color:
                            return False
        except (KeyError, IndexError, TypeError):
            print("Error: Cube state is malformed.")
            return False
            
        return True

    def display(self):
        """
        Prints a simple 2D text representation of the cube's state.
        This is very helpful for debugging.
        """
        print("       Up")
        for row in self.cube['U']:
            print("      " + " ".join(row))
        print("-" * 20)

        print(" Left   Front   Right   Back")
        for i in range(3):
            l_row = " ".join(self.cube['L'][i])
            f_row = " ".join(self.cube['F'][i])
            r_row = " ".join(self.cube['R'][i])
            b_row = " ".join(self.cube['B'][i])
            
            print(f"{l_row} | {f_row} | {r_row} | {b_row}")
        print("-" * 20)

        print("      Down")
        for row in self.cube['D']:
            print("      " + " ".join(row))
        print("--------------------\n")
        
    def get_state(self):
        """
        a deep copy of the current cube state.
        This is crucial for recursion, so you can pass a
        new state without modifying the old one.
        """
        return copy.deepcopy(self.cube)

    def set_state(self, new_state):
        """
        Sets the cube's state to a new state.
        """
        self.cube = new_state
    
    def get_hash(self):
        """
        Generates a unique string representation of the current cube state.
        Used as a key in the hash map for bidirectional search.
        """
        return "".join(
            sticker
            for face_name in ['U', 'D', 'F', 'B', 'R', 'L']
            for row in self.cube[face_name]
            for sticker in row
        )

    def load_state_from_string(self, state_str):
        """
        Parses a 54-character string into the cube state.
        Order: U, D, F, B, R, L (row by row, left to right).
        """
        if len(state_str) != 54:
            print("Error: State string must be exactly 54 characters long.")
            return False
            
        faces = ['U', 'D', 'F', 'B', 'R', 'L']
        idx = 0
        new_state = {}
        
        for face in faces:
            face_grid = []
            for r in range(3):
                row = []
                for c in range(3):
                    row.append(state_str[idx])
                    idx += 1
                face_grid.append(row)
            new_state[face] = face_grid
            
        self.cube = new_state
        return True

    def _rotate_face_clockwise(self, face_name):
        """
        Rotates the 9 stickers on a single given face clockwise.
        """
        face = self.cube[face_name]
        new_face = [['', '', ''], ['', '', ''], ['', '', '']]
        
        # Rotate Corners
        new_face[0][0] = face[2][0]
        new_face[0][2] = face[0][0]
        new_face[2][2] = face[0][2]
        new_face[2][0] = face[2][2]
        
        # Rotate Edges
        new_face[0][1] = face[1][0]
        new_face[1][2] = face[0][1]
        new_face[2][1] = face[1][2]
        new_face[1][0] = face[2][1]
        
        # Center remains same
        new_face[1][1] = face[1][1]
        
        self.cube[face_name] = new_face

    def _cycle_pieces(self, cycles):
        """
        Cycles pieces according to the provided coordinate lists.
        Each cycle is a list of 4 tuples: (face, row, col).
        The value moves from index 0 -> 1 -> 2 -> 3 -> 0.
        Used to update the stickers on the sides of the rotated face.
        """
        for p1, p2, p3, p4 in cycles:
            # Save current values
            v1 = self.cube[p1[0]][p1[1]][p1[2]]
            v2 = self.cube[p2[0]][p2[1]][p2[2]]
            v3 = self.cube[p3[0]][p3[1]][p3[2]]
            v4 = self.cube[p4[0]][p4[1]][p4[2]]
            
            # Assign new values (shifting clockwise)
            self.cube[p2[0]][p2[1]][p2[2]] = v1
            self.cube[p3[0]][p3[1]][p3[2]] = v2
            self.cube[p4[0]][p4[1]][p4[2]] = v3
            self.cube[p1[0]][p1[1]][p1[2]] = v4

    def move_f_clockwise(self):
        """
        Performs a clockwise rotation of the Front (F) face.
        This modifies the object's internal state.
        """
        self._rotate_face_clockwise('F')
        self._cycle_pieces([
            [('U', 2, 0), ('R', 0, 0), ('D', 0, 2), ('L', 2, 2)],
            [('U', 2, 1), ('R', 1, 0), ('D', 0, 1), ('L', 1, 2)],
            [('U', 2, 2), ('R', 2, 0), ('D', 0, 0), ('L', 0, 2)],
        ])

    def move_u_clockwise(self):
        """ Performs a clockwise rotation of the Up (U) face. """
        self._rotate_face_clockwise('U')
        self._cycle_pieces([
            [('F', 0, 0), ('R', 0, 0), ('B', 0, 0), ('L', 0, 0)],
            [('F', 0, 1), ('R', 0, 1), ('B', 0, 1), ('L', 0, 1)],
            [('F', 0, 2), ('R', 0, 2), ('B', 0, 2), ('L', 0, 2)],
        ])

    def move_d_clockwise(self):
        """ Performs a clockwise rotation of the Down (D) face. """
        self._rotate_face_clockwise('D')
        self._cycle_pieces([
            [('F', 2, 0), ('L', 2, 0), ('B', 2, 0), ('R', 2, 0)],
            [('F', 2, 1), ('L', 2, 1), ('B', 2, 1), ('R', 2, 1)],
            [('F', 2, 2), ('L', 2, 2), ('B', 2, 2), ('R', 2, 2)],
        ])

    def move_r_clockwise(self):
        """ Performs a clockwise rotation of the Right (R) face. """
        self._rotate_face_clockwise('R')
        self._cycle_pieces([
            [('U', 0, 2), ('B', 2, 0), ('D', 0, 2), ('F', 0, 2)],
            [('U', 1, 2), ('B', 1, 0), ('D', 1, 2), ('F', 1, 2)],
            [('U', 2, 2), ('B', 0, 0), ('D', 2, 2), ('F', 2, 2)],
        ])

    def move_l_clockwise(self):
        """ Performs a clockwise rotation of the Left (L) face. """
        self._rotate_face_clockwise('L')
        self._cycle_pieces([
            [('U', 0, 0), ('F', 0, 0), ('D', 0, 0), ('B', 2, 2)],
            [('U', 1, 0), ('F', 1, 0), ('D', 1, 0), ('B', 1, 2)],
            [('U', 2, 0), ('F', 2, 0), ('D', 2, 0), ('B', 0, 2)],
        ])
        
    def move_b_clockwise(self):
        """ Performs a clockwise rotation of the Back (B) face. """
        self._rotate_face_clockwise('B')
        self._cycle_pieces([
            [('U', 0, 0), ('L', 0, 0), ('D', 2, 2), ('R', 2, 2)],
            [('U', 0, 1), ('L', 1, 0), ('D', 2, 1), ('R', 1, 2)],
            [('U', 0, 2), ('L', 2, 0), ('D', 2, 0), ('R', 0, 2)],
        ])

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
        last_face = ''
        scramble_moves = []
        
        for _ in range(num_moves):
            while True:
                move_name = random.choice(move_names)
                current_face = move_name[0]
                
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

    def solve(self, max_moves=7, hash=False):
        """
        Attempts to solve the cube using Iterative Deepening
        Depth-First Search (IDDFS).
        
        If hash=True, it performs a bidirectional search (Meet-in-the-middle).
        1. Search forward from the scrambled state up to max_moves depth, storing states.
        2. Search backward from the solved state up to max_moves depth, checking for collisions.

        This guarantees the first solution found is the shortest.

        NOTE: This simple algorithm is EXTREMELY slow and will
        only find solutions for shallow scrambles (e.g., < 14 moves).
        """
        if hash:
            max_down_moves = max_moves
            print(f"Attempting to solve in a maximum of {max_moves*2} moves...")
        else:
            print(f"Attempting to solve in a maximum of {max_moves} moves...")
        
        initial_state_backup = self.get_state()

        # --- Phase 1: Forward Search ---
        for depth in range(max_moves + 1):
            print(f"  Searching at depth: {depth}")
            
            # hash_up=True means we store the end states of this search in self.all_hashes
            solution_path = self._recursive_search([], depth, hash)
            
            if solution_path is not None:
                print(f"\nSolution found in {len(solution_path)} moves!")
                print(f"  Path: {' '.join(solution_path)}")
                self.set_state(initial_state_backup)
                return solution_path
            
        # --- Phase 2: Backward Search (Bidirectional) ---
        if hash:
            self.set_state(self._get_solved_state_dict())
            for depth in range(max_down_moves + 1):
                print(f"  Searching at backwards depth: {depth}")

                # hash_down=True means we check if current state exists in self.all_hashes
                full_solution_path = self._recursive_search([], depth, False, hash)
                
                if full_solution_path is not None:
                    print(f"\nSolution found in {len(full_solution_path)} moves!")
                    print(f"  Path: {' '.join(full_solution_path)}")
                    self.set_state(initial_state_backup)
                    return full_solution_path
                
        self.set_state(initial_state_backup)

        if hash:
            print(f"\nNo solution found within {max_moves*2} moves.")
        else:
            print(f"\nNo solution found within {max_moves} moves.")
        return None

    def _recursive_search(self, path, depth_limit, hash_up=False, hash_down=False):
        """
        Recursive helper for the 'solve' method.
        Uses backtracking to avoid expensive deep copies.
        """
        # Base Case: Check if we found a solution or a meeting point
        if not hash_down:
            # Standard search: check if solved
            if self.is_solved():
                return path
        else:
            # Backward search: check if current state matches a state from forward search
            if self.get_hash() in self.all_hashes:
                return self.all_hashes[self.get_hash()] + [self._get_inverse_move_name(m) for m in reversed(path)]
            
        # Base Case: Depth limit reached
        if len(path) == depth_limit:
            if hash_up:
                # Store state for bidirectional search
                self.all_hashes[self.get_hash()] = path
            return None
            
        last_move = path[-1] if path else None
        
        for move_name, method_name in self.MOVE_MAPPING.items():
            
            if last_move:
                # 1. Don't immediately reverse the last move (e.g., F then F')
                if '_inv' in last_move and last_move.replace('_inv', '') == move_name:
                    continue
                if '_inv' not in last_move and last_move + '_inv' == move_name:
                    continue
                
                # 2. Don't do the same move 3 times (e.g., F F F is same as F')
                # This heuristic prunes paths like F F F, preferring F' instead.
                if (len(path) >= 2 and 
                    path[-1] == last_move and 
                    path[-2] == last_move and 
                    move_name == last_move):
                    continue
                
                # 3. Commutative moves pruning (e.g., treat F B and B F as same, pick one order)
                # This enforces an order on independent faces to reduce search space.
                current_face = move_name[0]
                last_face = last_move[0]
                
                if (last_face == 'B' and current_face == 'F') or \
                (last_face == 'L' and current_face == 'R') or \
                (last_face == 'D' and current_face == 'U'):
                    continue

            # Apply move
            move_function = getattr(self, method_name)
            move_function()

            # Recurse
            result = self._recursive_search(path + [move_name], depth_limit, hash_up, hash_down)
            
            if result is not None:
                return result
            
            # Backtrack (undo move)
            inverse_move = self._get_inverse_move_name(move_name)
            inverse_method = self.MOVE_MAPPING[inverse_move]
            getattr(self, inverse_method)()
            
        return None
