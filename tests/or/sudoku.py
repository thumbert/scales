"""Google OR-Tools solution to a Sudoku problem."""
import sys
import time
from ortools.sat.python import cp_model


class SudokuSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, values: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__values = values
        self.__solution_count = 0
        self.__start_time = time.time()

    @property
    def solution_count(self) -> int:
        return self.__solution_count

    def on_solution_callback(self):
        current_time = time.time()
        print(
            f"Solution {self.__solution_count}, "
            f"time = {current_time - self.__start_time} s"
        )
        self.__solution_count += 1

        solution = self.response_proto.solution
        for i in range(9):
            print("".join(str(solution[i * 9 + j]) for j in range(9)))

        print()

puzzle1 = """
.......1.
4........
.2.......
....5.4.7
..8...3..
..1.9....
3..4..2..
.5.1.....
...8.6...
"""

puzzle2 = """
53..7....
6..195...
.98....6.
8...6...3
4..8.3..1
7...2...6
.6....28.
...419..5
....8..79
"""



def read_board(puzzle: str, board_size: int) -> list[int]:
    """Reads a Sudoku board from a string."""
    board = []
    for line in puzzle.strip().splitlines():
        for char in line.strip():
            if char == '.':
                board.append(0)  # Empty cell
            else:
                board.append(int(char))  # Given value
    assert len(board) == board_size * board_size, "Board size mismatch"
    return board

# See https://users.aalto.fi/~tjunttil/2020-DP-AUT/notes-sat/overview.html
def main(board_size: int) -> None:

    input = read_board(puzzle2, board_size)
    # print(input)

    # Creates the solver.
    model = cp_model.CpModel()

    # Creates the variables.
    # There are `board_size`^2 number of variables, one for each cell.
    # The value of each variable is the number in that cell, from 1 to `board_size`.
    # Using matrix notation, the variable at (x, y) is `values[x*board_size + y]` 
    # are stored in row-major notation.
    #
    values = []
    for y in range(board_size):
        for x in range(board_size):
            values.append(model.new_int_var(1, board_size, f"v_{x}_{y}"))

    # Creates the constraints:
    # All cells in a row must be different.
    for x in range(board_size):
        model.add_all_different(values[y + x*board_size] for y in range(board_size))    

    # All cells in a column must be different.
    for y in range(board_size):
        model.add_all_different(values[x*board_size + y] for x in range(board_size))    

    # All cells in a block must be different.
    for block_x in range(0, board_size, 3):
        for block_y in range(0, board_size, 3):
            model.add_all_different(
                values[(block_x + dx) + (block_y + dy) * board_size]
                for dx in range(3) for dy in range(3)
            )

    # Add the given values 
    for i in range(len(input)):
        if input[i] != 0:
            # The cell at (x, y) corresponds to input[i] and is already set.
            # We add a constraint that this cell must equal the given value.
            model.add(values[i] == input[i])


    # Solve the model.
    solver = cp_model.CpSolver()
    # solver.parameters.enumerate_all_solutions = True
    solution_printer = SudokuSolutionPrinter(values)
    solver.solve(model, solution_printer)

    # Statistics.
    print("\nStatistics")
    print(f"  conflicts      : {solver.num_conflicts}")
    print(f"  branches       : {solver.num_branches}")
    print(f"  wall time      : {solver.wall_time} s")
    print(f"  solutions found: {solution_printer.solution_count}")


if __name__ == "__main__":
    # Solve the classic 9x9 Sudoku.
    size = 9
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    main(size)