# Solve NYTimes "Pips" puzzle using CP-SAT solver.
# See https://www.nytimes.com/puzzles/pips


from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, CpSolver, IntVar, LinearExpr
from typing import Optional
import math
import sys
import time

from abc import ABC, abstractmethod
from typing import List, Dict


class ConstraintState:
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    UNDECIDABLE = "undecidable"


class Constraint(ABC):
    def __init__(self, cell_ids: List[str]):
        self.cell_ids = cell_ids

    @abstractmethod
    def is_satisfied(self, cells: Dict[str, int]) -> str:
        pass


class ConstraintEqual(Constraint):
    def is_satisfied(self, filled_cells: Dict[str, int]) -> str:
        if any(cell_id not in filled_cells for cell_id in self.cell_ids):
            return ConstraintState.UNDECIDABLE
        values = {filled_cells[cell_id] for cell_id in self.cell_ids}
        return (
            ConstraintState.SATISFIED
            if len(values) == 1
            else ConstraintState.UNSATISFIED
        )


class ConstraintLessThan(Constraint):
    def __init__(self, cell_ids: List[str], total: int):
        super().__init__(cell_ids)
        self.total = total

    def is_satisfied(self, filled_cells: Dict[str, int], total: int) -> str:
        if any(cell_id not in filled_cells for cell_id in self.cell_ids):
            return ConstraintState.UNDECIDABLE
        total_sum = sum(filled_cells[cell_id] for cell_id in self.cell_ids)
        return (
            ConstraintState.SATISFIED
            if total_sum < total
            else ConstraintState.UNSATISFIED
        )


class ConstraintGreaterThan(Constraint):
    def __init__(self, cell_ids: List[str], total: int):
        super().__init__(cell_ids)
        self.total = total

    def is_satisfied(self, filled_cells: Dict[str, int], total: int) -> str:
        if any(cell_id not in filled_cells for cell_id in self.cell_ids):
            return ConstraintState.UNDECIDABLE
        total_sum = sum(filled_cells[cell_id] for cell_id in self.cell_ids)
        return (
            ConstraintState.SATISFIED
            if total_sum > total
            else ConstraintState.UNSATISFIED
        )


class ConstraintNotEqual(Constraint):
    def is_satisfied(self, filled_cells: Dict[str, int]) -> str:
        if any(cell_id not in filled_cells for cell_id in self.cell_ids):
            return ConstraintState.UNDECIDABLE
        values = {filled_cells[cell_id] for cell_id in self.cell_ids}
        return (
            ConstraintState.SATISFIED
            if len(values) == len(self.cell_ids)
            else ConstraintState.UNSATISFIED
        )


class ConstraintSum(Constraint):
    def __init__(self, cell_ids: List[str], total: int):
        super().__init__(cell_ids)
        self.total = total

    def is_satisfied(self, filled_cells: Dict[str, int]) -> str:
        if any(cell_id not in filled_cells for cell_id in self.cell_ids):
            return ConstraintState.UNDECIDABLE
        total_sum = sum(filled_cells[cell_id] for cell_id in self.cell_ids)
        return (
            ConstraintState.SATISFIED
            if total_sum == self.total
            else ConstraintState.UNSATISFIED
        )


class PipsSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(
        self,
        values: list[cp_model.IntVar],
        dominoes: set[tuple[int, int]],
        links: dict[str, list[str]],
        configurations: list[list[str]],
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__values = values
        self.__dominoes = dominoes
        self.__links = links
        self.__configurations = configurations
        self.__solution_count = 0
        self.__start_time = time.time()

    @property
    def solution_count(self) -> int:
        return self.__solution_count

    def on_solution_callback(self):
        current_time = time.time()

        solution = self.response_proto.solution
        N = len(solution) // 7
        nodes = list(self.__links.keys())
        indices = [i for i, val in enumerate(solution) if val == 1]
        filled_cells = {nodes[i // 7]: i % 7 for i in indices}

        # Check if the solution matches one of the valid configurations.
        for config in self.__configurations:
            stones: set[tuple[int, int]] = set()
            for cell1, cell2 in [link.strip("()").split("-") for link in config]:
                val1 = filled_cells[cell1]
                val2 = filled_cells[cell2]
                stone = (min(val1, val2), max(val1, val2))
                stones.add(stone)
        
            if stones == self.__dominoes:
                print(
                    f"Solution {self.__solution_count}, "
                    f"time = {current_time - self.__start_time} s"
                )
                self.__solution_count += 1
                print(f"Solution: {filled_cells}")
                print(f"Configuration: {config}")


class PipsLinksSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(
        self,
        values: list[cp_model.IntVar],
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__values = values
        self.__solution_count = 0
        self.__configs: list[list[str]] = []

    @property
    def solution_count(self) -> int:
        return self.__solution_count

    @property
    def configurations(self) -> list[list[str]]:
        return self.__configs

    def on_solution_callback(self):
        solution = self.response_proto.solution
        indices = [i for i, val in enumerate(solution) if val == 1]
        self.__solution_count += 1
        self.__configs.append([self.__values[i].name for i in indices])

# Generate all the possible arrangements of dominoes.
# For small and medium puzzles, it ends up being a small number of possibilities,
# usually less than 10.
# For large puzzles, the number of possible arrangements are usually in the hundreds.
#
def solve_puzzle_links(puzzle):
    model = CpModel()

    # Create the variables.  Model each link as a variable that can take value 0 or 1.
    # Value 0 means the two nodes are not linked, value 1 means they are linked.
    x: List[IntVar] = []
    for cell, neighbors in puzzle["links"].items():
        for neighbor in neighbors:
            if cell < neighbor:  # to avoid duplicates
                x.append(model.new_int_var(0, 1, f"({cell}-{neighbor})"))
    N = len(x)
    print(f"Number of links: {N}")

    # Each node is linked to exactly one other node.
    for cell, neighbors in puzzle["links"].items():
        model.add(
            sum(
                [
                    x[i]
                    for i in range(N)
                    if x[i].name
                    in [
                        f"({cell}-{n})" if cell < n else f"({n}-{cell})"
                        for n in neighbors
                    ]
                ]
            )
            == 1
        )

    solver = CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solution_printer = PipsLinksSolutionPrinter(x)
    status = solver.solve(model, solution_printer)
    if status != cp_model.OPTIMAL:
        print("Not all link solutions were found!")

    # Statistics.
    print("\nStatistics:")
    print(f"  conflicts      : {solver.num_conflicts}")
    print(f"  branches       : {solver.num_branches}")
    print(f"  wall time      : {solver.wall_time} s")
    print(f"  solutions found: {solution_printer.solution_count}")
    print(f"  configurations :")
    for config in solution_printer.configurations:
        print(f"    {config}")

    return solution_printer.configurations



def solve_puzzle_values(puzzle, configurations: list[list[str]]):
    model = CpModel()

    N = len(puzzle["dominoes"]) * 2
    nodes = list(puzzle["links"].keys())
    assert len(nodes) == N

    # Create the variables.  There are 7*N total variables.  Will model them as a
    # flat list of boolean variables.
    # The variables [0-7) are for the first node,  [7-14) for the second, and so on.
    # The variable with index 0 equals true if the first node == 0, and equals false if
    # the first node != 0.
    # The variable with index 1 equals true if the first node == 1, and equals false if
    # the first node != 1, etc.
    x: List[IntVar] = []
    for i in range(N * 7):
        x.append(model.new_int_var(0, 1, f"{nodes[i // 7]}_{i % 7}"))

    # Each node takes exactly one value.
    for i in range(N):
        model.add(sum(x[i * 7 + j] for j in range(7)) == 1)

    # Add the constraints
    for constraint in puzzle["constraints"]:
        if isinstance(constraint, ConstraintEqual):
            for i in range(7):
                for j in range(len(constraint.cell_ids) - 1):
                    cell1 = constraint.cell_ids[j]
                    cell2 = constraint.cell_ids[j + 1]
                    index1 = nodes.index(cell1)
                    index2 = nodes.index(cell2)
                    model.add(x[index1 * 7 + i] == x[index2 * 7 + i])
        elif isinstance(constraint, ConstraintSum):
            model.add(
                sum(
                    sum(j * x[nodes.index(cell_id) * 7 + j] for j in range(7))
                    for cell_id in constraint.cell_ids
                )
                == constraint.total
            )
        elif isinstance(constraint, ConstraintLessThan):
            model.add(
                sum(
                    sum(j * x[nodes.index(cell_id) * 7 + j] for j in range(7))
                    for cell_id in constraint.cell_ids
                )
                < constraint.total
            )
        elif isinstance(constraint, ConstraintGreaterThan):
            model.add(
                sum(
                    sum(j * x[nodes.index(cell_id) * 7 + j] for j in range(7))
                    for cell_id in constraint.cell_ids
                )
                > constraint.total
            )
        elif isinstance(constraint, ConstraintNotEqual):
            for v1 in range(7):
                model.add(
                    sum(
                        x[nodes.index(cell_id) * 7 + v1]
                        for cell_id in constraint.cell_ids
                    )
                    <= 1
                )
        else:
            raise ValueError(f"Unknown constraint type: {type(constraint)}")

    # Impose the constraints from the domino counts
    counts = {n: sum(t.count(n) for t in puzzle["dominoes"]) for n in range(7)}
    for n in range(7):
        model.add(sum(x[7 * j + n] for j in range(N)) == counts[n])

    # Create the solver and solve.
    print("\n\nFinding the solution that matches the dominoes configurations...")
    solver = CpSolver()
    solution_printer = PipsSolutionPrinter(
        x, puzzle["dominoes"], puzzle["links"], configurations
    )
    # status = solver.Solve(model, solution_printer)
    status = solver.SearchForAllSolutions(model, solution_printer)
    if status != cp_model.OPTIMAL:
        print("Not all values solutions were found!")

    # Statistics.
    print("Statistics")
    print(f"  conflicts      : {solver.num_conflicts}")
    print(f"  branches       : {solver.num_branches}")
    print(f"  wall time      : {solver.wall_time} s")
    print(f"  solutions found: {solution_printer.solution_count}")

    return (model, x)



# https://pedtsr.ca/2024/solving-domino-fit-using-constraint-programming.html


# easy puzzle 2025-09-16
puzzle1 = {
    "dominoes": {(2, 4), (1, 6), (4, 4), (1, 1), (1, 5), (2, 6)},
    "constraints": [
        ConstraintEqual(["b", "c", "d", "e"]),
        ConstraintSum(["f", "g"], 11),
        ConstraintSum(["h", "i"], 4),
    ],
    "links": {
        "a": ["b", "l"],
        "b": ["a", "c"],
        "c": ["b", "d"],
        "d": ["c", "e"],
        "e": ["d", "f"],
        "f": ["e", "g"],
        "g": ["f", "h"],
        "h": ["g", "i"],
        "i": ["h", "j"],
        "j": ["i", "k"],
        "k": ["j", "l"],
        "l": ["k", "a"],
    },
    "solution": {
        "values": [6, 1, 1, 1, 1, 5, 6, 2, 2, 4, 4, 4],
        "config": [
            "(a-b)",
            "(c-d)",
            "(e-f)",
            "(f-g)",
            "(h-i)",
            "(j-k)",
            "(k-l)",
        ],
    },
}

# hard puzzle 2025-09-21
puzzle2 = {
    "dominoes": {
        (1, 2),
        (0, 6),
        (5, 5),
        (1, 3),
        (4, 4),
        (1, 6),
        (0, 0),
        (3, 4),
        (2, 5),
        (3, 6),
        (6, 6),
        (4, 5),
    },
    "constraints": [
        ConstraintEqual(["a", "b"]),
        ConstraintSum(["c"], 3),
        ConstraintSum(["d"], 5),
        ConstraintSum(["e"], 1),
        ConstraintSum(["f", "k", "q"], 18),
        ConstraintSum(["g"], 3),
        ConstraintLessThan(["h"], 4),
        ConstraintNotEqual(["i", "j", "n", "o", "p"]),
        ConstraintGreaterThan(["m"], 4),
        ConstraintSum(["l", "r"], 10),
        ConstraintSum(["s"], 2),
        ConstraintSum(["t", "w", "x"], 0),
        ConstraintSum(["u"], 4),
        ConstraintSum(["v"], 1),
    ],
    "links": {
        "a": ["b", "d"],
        "b": ["a", "e"],
        "c": ["d", "h"],
        "d": ["a", "c", "e", "i"],
        "e": ["b", "d", "f", "j"],
        "f": ["e", "k"],
        "g": ["h", "m"],
        "h": ["c", "g", "i", "n"],
        "i": ["d", "h", "j", "o"],
        "j": ["e", "i", "k", "p"],
        "k": ["f", "j", "l", "q"],
        "l": ["k", "r"],
        "m": ["g", "n"],
        "n": ["h", "m", "o", "s"],
        "o": ["i", "n", "p", "t"],
        "p": ["j", "o", "q", "u"],
        "q": ["k", "p", "r", "v"],
        "r": ["l", "q"],
        "s": ["n", "t"],
        "t": ["o", "s", "u", "w"],
        "u": ["p", "t", "v", "x"],
        "v": ["q", "u"],
        "w": ["t", "x"],
        "x": ["u", "w"],
    },
    "solution": {
        "values": [
            4,
            4,
            3,
            5,
            1,
            6,
            3,
            1,
            4,
            2,
            6,
            5,
            6,
            5,
            6,
            3,
            6,
            5,
            2,
            0,
            4,
            1,
            0,
            0,
        ],
        "config": [
            "(a-b)",
            "(c-h)",
            "(d-i)",
            "(e-j)",
            "(f-k)",
            "(g-m)",
            "(l-r)",
            "(n-s)",
            "(o-t)",
            "(p-u)",
            "(q-v)",
            "(w-x)",
        ],
    },
}



def main() -> None:
    # Choose the puzzle to solve:
    puzzle = puzzle1

    # First, generate all the ways the dominoes can fit on the grid.
    configurations = solve_puzzle_links(puzzle)

    # Second, solve the puzzle using the dominoes values, by imposing the
    # numerical constraints on the values of the dominoes and then finding
    # the solution that matches one of the valid configurations.
    solve_puzzle_values(puzzle, configurations)


if __name__ == "__main__":
    main()
