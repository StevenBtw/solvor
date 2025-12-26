"""
Bin Packing Example

Pack items into minimum number of bins without exceeding capacity.
"""

from solvor import solve_binpack

# Item sizes to pack
item_sizes = [4, 8, 5, 1, 7, 6, 1, 4, 2, 3]
bin_capacity = 10

result = solve_binpack(item_sizes, bin_capacity, algorithm="best-fit-decreasing")
print(f"Bin assignments: {result.solution}")
print(f"Number of bins used: {int(result.objective)}")
