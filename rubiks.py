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

    def _rotate_face_clockwise(self, face_name):
        """
        Rotates the 9 stickers on a single given face clockwise.
        """
        face = self.cube[face_name]
        new_face = [['', '', ''], ['', '', ''], ['', '', '']]
        
        new_face[0][0] = face[2][0]
        new_face[0][2] = face[0][0]
        new_face[2][2] = face[0][2]
        new_face[2][0] = face[2][2]
        
        new_face[0][1] = face[1][0]
        new_face[1][2] = face[0][1]
        new_face[2][1] = face[1][2]
        new_face[1][0] = face[2][1]
        
        new_face[1][1] = face[1][1]
        
        self.cube[face_name] = new_face

    def move_f_clockwise(self):
        """
        Performs a clockwise rotation of the Front (F) face.
        This modifies the object's internal state.
        """
        
        self._rotate_face_clockwise('F')
        
        temp_up_row = self.cube['U'][2][:]
        
        self.cube['U'][2][0] = self.cube['L'][2][2]
        self.cube['U'][2][1] = self.cube['L'][1][2]
        self.cube['U'][2][2] = self.cube['L'][0][2]
        
        self.cube['L'][0][2] = self.cube['D'][0][0]
        self.cube['L'][1][2] = self.cube['D'][0][1]
        self.cube['L'][2][2] = self.cube['D'][0][2]
        
        self.cube['D'][0][0] = self.cube['R'][2][0]
        self.cube['D'][0][1] = self.cube['R'][1][0]
        self.cube['D'][0][2] = self.cube['R'][0][0]
        
        self.cube['R'][0][0] = temp_up_row[0]
        self.cube['R'][1][0] = temp_up_row[1]
        self.cube['R'][2][0] = temp_up_row[2]

    def move_u_clockwise(self):
        """ Performs a clockwise rotation of the Up (U) face. """
        
        self._rotate_face_clockwise('U')
        
        temp_row = self.cube['F'][0][:]
        self.cube['F'][0] = self.cube['L'][0][:]
        self.cube['L'][0] = self.cube['B'][0][:]
        self.cube['B'][0] = self.cube['R'][0][:]
        self.cube['R'][0] = temp_row

    def move_d_clockwise(self):
        """ Performs a clockwise rotation of the Down (D) face. """
        
        self._rotate_face_clockwise('D')
        
        temp_row = self.cube['F'][2][:]
        self.cube['F'][2] = self.cube['R'][2][:]
        self.cube['R'][2] = self.cube['B'][2][:]
        self.cube['B'][2] = self.cube['L'][2][:]
        self.cube['L'][2] = temp_row

    def move_r_clockwise(self):
        """ Performs a clockwise rotation of the Right (R) face. """
        
        self._rotate_face_clockwise('R')
        
        temp_col = [self.cube['U'][0][2], self.cube['U'][1][2], self.cube['U'][2][2]]
        
        self.cube['U'][0][2] = self.cube['F'][0][2]
        self.cube['U'][1][2] = self.cube['F'][1][2]
        self.cube['U'][2][2] = self.cube['F'][2][2]
        
        self.cube['F'][0][2] = self.cube['D'][0][2]
        self.cube['F'][1][2] = self.cube['D'][1][2]
        self.cube['F'][2][2] = self.cube['D'][2][2]

        self.cube['D'][0][2] = self.cube['B'][2][0]
        self.cube['D'][1][2] = self.cube['B'][1][0]
        self.cube['D'][2][2] = self.cube['B'][0][0]
        
        self.cube['B'][0][0] = temp_col[2]
        self.cube['B'][1][0] = temp_col[1]
        self.cube['B'][2][0] = temp_col[0]

    def move_l_clockwise(self):
        """ Performs a clockwise rotation of the Left (L) face. """
        
        self._rotate_face_clockwise('L')
        
        temp_col = [self.cube['U'][0][0], self.cube['U'][1][0], self.cube['U'][2][0]]
        
        self.cube['U'][0][0] = self.cube['B'][2][2]
        self.cube['U'][1][0] = self.cube['B'][1][2]
        self.cube['U'][2][0] = self.cube['B'][0][2]

        self.cube['B'][0][2] = self.cube['D'][2][0]
        self.cube['B'][1][2] = self.cube['D'][1][0]
        self.cube['B'][2][2] = self.cube['D'][0][0]

        self.cube['D'][0][0] = self.cube['F'][0][0]
        self.cube['D'][1][0] = self.cube['F'][1][0]
        self.cube['D'][2][0] = self.cube['F'][2][0]

        self.cube['F'][0][0] = temp_col[0]
        self.cube['F'][1][0] = temp_col[1]
        self.cube['F'][2][0] = temp_col[2]
        
    def move_b_clockwise(self):
        """ Performs a clockwise rotation of the Back (B) face. """
        
        self._rotate_face_clockwise('B')
        
        temp_row = self.cube['U'][0][:]

        self.cube['U'][0][0] = self.cube['R'][2][2]
        self.cube['U'][0][1] = self.cube['R'][1][2]
        self.cube['U'][0][2] = self.cube['R'][0][2]

        self.cube['R'][0][2] = self.cube['D'][2][0]
        self.cube['R'][1][2] = self.cube['D'][2][1]
        self.cube['R'][2][2] = self.cube['D'][2][2]

        self.cube['D'][2][0] = self.cube['L'][2][0]
        self.cube['D'][2][1] = self.cube['L'][1][0]
        self.cube['D'][2][2] = self.cube['L'][0][0]

        self.cube['L'][0][0] = temp_row[0]
        self.cube['L'][1][0] = temp_row[1]
        self.cube['L'][2][0] = temp_row[2]

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
        
        initial_state_backup = self.get_state()

        for depth in range(max_moves + 1):
            print(f"  Searching at depth: {depth}")
            
            solution_path = self._recursive_search([], depth)
            
            if solution_path is not None:
                print(f"\nSolution found in {len(solution_path)} moves!")
                print(f"  Path: {' '.join(solution_path)}")
                self.set_state(initial_state_backup)
                return solution_path
                
        self.set_state(initial_state_backup)
        print(f"\nNo solution found within {max_moves} moves.")
        return None

    def _recursive_search(self, path, depth_limit):
        """
        Recursive helper for the 'solve' method.
        Uses backtracking to avoid expensive deep copies.
        """
        
        if self.is_solved():
            return path
            
        if len(path) == depth_limit:
            return None
            
        last_move = path[-1] if path else None
        
        for move_name, method_name in self.MOVE_MAPPING.items():
            
            if last_move:
                if '_inv' in last_move and last_move.replace('_inv', '') == move_name:
                    continue
                if '_inv' not in last_move and last_move + '_inv' == move_name:
                    continue
                
                if (len(path) >= 2 and 
                    path[-1] == last_move and 
                    path[-2] == last_move and 
                    move_name == last_move):
                    continue
                
                current_face = move_name[0]
                last_face = last_move[0]
                
                if (last_face == 'B' and current_face == 'F') or \
                   (last_face == 'L' and current_face == 'R') or \
                   (last_face == 'D' and current_face == 'U'):
                    continue

            move_function = getattr(self, method_name)
            move_function()
            
            result = self._recursive_search(path + [move_name], depth_limit)
            
            if result is not None:
                return result
            
            inverse_move = self._get_inverse_move_name(move_name)
            inverse_method = self.MOVE_MAPPING[inverse_move]
            getattr(self, inverse_method)()
        
        return None
