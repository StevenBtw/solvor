r"""
MILP Solver, linear optimization with integer constraints.

Linear programming with integer constraints. The workhorse of discrete
optimization: diet problems, scheduling, set covering, facility location.

    from solvor.milp import solve_milp

    # minimize c @ x, subject to A @ x <= b, x >= 0, some x integer
    result = solve_milp(c, A, b, integers=[0, 2])
    result = solve_milp(c, A, b, integers=[0, 1], minimize=False)  # maximize

    # warm start from previous solution (prunes search tree)
    result = solve_milp(c, A, b, integers=[0, 2], warm_start=previous.solution)

    # find multiple solutions (result.solutions contains all found)
    result = solve_milp(c, A, b, integers=[0, 2], solution_limit=5)

How it works: branch and bound using simplex as subroutine. Solves LP relaxations
(ignoring integer constraints), branches on fractional values, prunes subtrees
that can't beat current best. Best-first search prioritizes promising branches.

Use this for:

- Linear objectives with integer constraints
- Diet/blending, scheduling, set covering
- Facility location, power grid design
- When you need proven optimal values

Parameters:

    c: objective coefficients (minimize c @ x)
    A: constraint matrix (A @ x <= b)
    b: constraint bounds
    integers: indices of integer-constrained variables
    minimize: True for min, False for max (default: True)
    warm_start: initial solution to prune search tree
    solution_limit: find multiple solutions (default: 1)
    heuristics: rounding + local search heuristics (default: True)
    lns_iterations: LNS improvement passes, 0 = off (default: 0)
    lns_destroy_frac: fraction of variables to unfix per LNS iteration (default: 0.3)
    seed: random seed for LNS reproducibility
    max_nodes: branch-and-bound node limit (default: 100000)
    gap_tol: optimality gap tolerance (default: 1e-6)

CP is more expressive for logical constraints. SAT handles pure boolean.
For continuous-only problems, use simplex directly.
"""

from collections.abc import Sequence
from heapq import heappop, heappush
from math import ceil, floor
from random import Random
from typing import NamedTuple

from solvor.lns import lns as _lns
from solvor.simplex import Status as LPStatus
from solvor.simplex import solve_lp
from solvor.types import Result, Status
from solvor.utils import check_integers_valid, check_matrix_dims, warn_large_coefficients

__all__ = ["solve_milp"]


# I deliberately picked NamedTuple over dataclass for performance
class Node(NamedTuple):
    bound: float
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    depth: int


def solve_milp(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    integers: Sequence[int],
    *,
    minimize: bool = True,
    eps: float = 1e-6,
    max_iter: int = 10_000,
    max_nodes: int = 100_000,
    gap_tol: float = 1e-6,
    warm_start: Sequence[float] | None = None,
    solution_limit: int = 1,
    heuristics: bool = True,
    lns_iterations: int = 0,
    lns_destroy_frac: float = 0.3,
    seed: int | None = None,
) -> Result:
    n = len(c)
    check_matrix_dims(c, A, b)
    check_integers_valid(integers, n)
    warn_large_coefficients(A)

    int_set = set(integers)
    total_iters = 0

    lower = [0.0] * n
    upper = [float("inf")] * n

    root_result = _solve_node(c, A, b, lower, upper, minimize, eps, max_iter)
    total_iters += root_result.iterations

    if root_result.status == LPStatus.INFEASIBLE:
        return Result(None, float("inf") if minimize else float("-inf"),
                     0, total_iters, Status.INFEASIBLE)

    if root_result.status == LPStatus.UNBOUNDED:
        return Result(None, float("-inf") if minimize else float("inf"),
                     0, total_iters, Status.UNBOUNDED)

    best_solution, best_obj = None, float("inf") if minimize else float("-inf")
    sign = 1 if minimize else -1
    all_solutions: list[tuple] = []

    # Use warm start as initial incumbent if provided and feasible
    if warm_start is not None:
        ws = tuple(warm_start)
        if len(ws) == n and _is_feasible(ws, A, b, int_set, eps):
            best_obj = sum(c[j] * ws[j] for j in range(n))
            best_solution = ws
            all_solutions.append(ws)

    frac_var = _most_fractional(root_result.solution, int_set, eps)

    if frac_var is None:
        return Result(root_result.solution, root_result.objective, 1, total_iters)

    # Check if LP relaxation suggests binary (values in [0,1])
    looks_binary = all(-eps <= root_result.solution[j] <= 1 + eps for j in int_set)

    # Only tighten bounds if explicit x_j <= 1 constraints exist
    if looks_binary and _detect_binary(A, b, int_set, n, eps):
        for j in int_set:
            lower[j] = max(lower[j], 0.0)
            upper[j] = min(upper[j], 1.0)

    # Run heuristics if LP looks binary (even without explicit constraints)
    if heuristics and looks_binary and best_solution is None:
        rounded = _round_binary(root_result.solution, int_set, c, A, b, minimize, eps)
        if rounded is not None:
            best_obj = sum(c[j] * rounded[j] for j in range(n))
            best_solution = rounded
            all_solutions.append(rounded)

    # LNS improvement for binary problems
    if heuristics and looks_binary and lns_iterations > 0 and best_solution is not None:
        rng = Random(seed)
        improved, iters = _lns_improve(
            best_solution, c, A, b, int_set, minimize, eps, max_iter,
            lns_iterations, lns_destroy_frac, rng
        )
        total_iters += iters
        if improved is not None:
            improved_obj = sum(c[j] * improved[j] for j in range(n))
            if (minimize and improved_obj < best_obj) or (not minimize and improved_obj > best_obj):
                best_solution, best_obj = improved, improved_obj
                if improved not in all_solutions:
                    all_solutions.append(improved)

    tree: list[tuple[float, int, Node]] = []
    counter = 0
    root_bound = sign * root_result.objective
    heappush(tree, (root_bound, counter, Node(root_bound, tuple(lower), tuple(upper), 0)))
    counter += 1
    nodes_explored = 0

    while tree and nodes_explored < max_nodes:
        node_bound, _, node = heappop(tree)

        # Prune if can't improve
        if best_solution is not None and node_bound >= sign * best_obj - eps:
            continue

        result = _solve_node(c, A, b, node.lower, node.upper, minimize, eps, max_iter)
        total_iters += result.iterations
        nodes_explored += 1

        if result.status != LPStatus.OPTIMAL:
            continue

        if best_solution is not None and sign * result.objective >= sign * best_obj - eps:
            continue

        frac_var = _most_fractional(result.solution, int_set, eps)

        if frac_var is None:
            # Found an integer-feasible solution
            sol = tuple(result.solution)
            sol_obj = result.objective

            # Collect solution if within limit
            if solution_limit > 1 and sol not in all_solutions:
                all_solutions.append(sol)
                if len(all_solutions) >= solution_limit:
                    return Result(best_solution or sol,
                                 best_obj if best_solution else sol_obj,
                                 nodes_explored, total_iters, Status.FEASIBLE,
                                 solutions=tuple(all_solutions))

            if sign * sol_obj < sign * best_obj:
                best_solution, best_obj = sol, sol_obj
                gap = _compute_gap(best_obj, node_bound / sign if node_bound != 0 else 0)
                if gap < gap_tol and solution_limit == 1:
                    return Result(best_solution, best_obj, nodes_explored, total_iters)

            continue

        val = result.solution[frac_var]
        child_bound = sign * result.objective

        lower_left, upper_left = list(node.lower), list(node.upper)
        upper_left[frac_var] = floor(val)
        heappush(tree, (child_bound, counter,
                       Node(child_bound, tuple(lower_left), tuple(upper_left), node.depth + 1)))
        counter += 1

        lower_right, upper_right = list(node.lower), list(node.upper)
        lower_right[frac_var] = ceil(val)
        heappush(tree, (child_bound, counter,
                       Node(child_bound, tuple(lower_right), tuple(upper_right), node.depth + 1)))
        counter += 1

    if best_solution is None:
        return Result(None, float("inf") if minimize else float("-inf"),
                     nodes_explored, total_iters, Status.INFEASIBLE)

    status = Status.OPTIMAL if not tree else Status.FEASIBLE
    if solution_limit > 1 and all_solutions:
        return Result(best_solution, best_obj, nodes_explored, total_iters, status,
                     solutions=tuple(all_solutions))
    return Result(best_solution, best_obj, nodes_explored, total_iters, status)


def _solve_node(c, A, b, lower, upper, minimize, eps, max_iter):
    # Substitute fixed variables to reduce problem size
    n = len(c)
    fixed = {}
    free_vars = []

    for j in range(n):
        lo, hi = lower[j], upper[j]
        if hi < lo - eps:
            return Result(None, float("inf") if minimize else float("-inf"),
                         0, 0, LPStatus.INFEASIBLE)
        if hi - lo < eps:
            fixed[j] = lo
        else:
            free_vars.append(j)

    if not free_vars:
        obj = sum(c[j] * fixed[j] for j in fixed)
        sol = [fixed.get(j, 0.0) for j in range(n)]
        for i, row in enumerate(A):
            lhs = sum(row[j] * sol[j] for j in range(n))
            if lhs > b[i] + eps:
                return Result(None, float("inf") if minimize else float("-inf"),
                             0, 0, LPStatus.INFEASIBLE)
        return Result(tuple(sol), obj, 0, 0, LPStatus.OPTIMAL)

    # Build reduced problem
    n_free = len(free_vars)
    A_red, b_red = [], []

    for i, row in enumerate(A):
        fixed_contrib = sum(row[j] * fixed[j] for j in fixed)
        new_rhs = b[i] - fixed_contrib
        A_red.append([row[j] for j in free_vars])
        b_red.append(new_rhs)

    # Only add non-trivial bounds
    for j_new, j_old in enumerate(free_vars):
        lo, hi = lower[j_old], upper[j_old]
        if lo > eps:
            row = [0.0] * n_free
            row[j_new] = -1.0
            A_red.append(row)
            b_red.append(-lo)
        if hi < float("inf"):
            row = [0.0] * n_free
            row[j_new] = 1.0
            A_red.append(row)
            b_red.append(hi)

    c_red = [c[j] for j in free_vars]
    fixed_obj = sum(c[j] * fixed[j] for j in fixed)

    result = solve_lp(c_red, A_red, b_red, minimize=minimize, eps=eps, max_iter=max_iter)

    if result.status != LPStatus.OPTIMAL:
        return result

    # Reconstruct full solution
    full_sol = [0.0] * n
    for j in fixed:
        full_sol[j] = fixed[j]
    for j_new, j_old in enumerate(free_vars):
        full_sol[j_old] = result.solution[j_new]

    return Result(tuple(full_sol), result.objective + fixed_obj,
                 result.iterations, result.iterations, result.status)


def _most_fractional(solution, int_set, eps):
    best_var, best_frac = None, 0.0
    for j in int_set:
        val = solution[j]
        frac = abs(val - round(val))
        if frac > eps and frac > best_frac:
            best_var, best_frac = j, frac
    return best_var


def _compute_gap(best_obj, bound):
    if abs(best_obj) < 1e-10:
        return abs(best_obj - bound)
    return abs(best_obj - bound) / abs(best_obj)


def _detect_binary(A, b, int_set, n, eps):
    """Check if integer variables have explicit x_j <= 1 constraints."""
    bounded = set()
    for i, row in enumerate(A):
        if abs(b[i] - 1.0) > eps:
            continue
        # Check if row is a single-variable upper bound: x_j <= 1
        nz = [(j, row[j]) for j in range(n) if abs(row[j]) > eps]
        if len(nz) == 1:
            j, coef = nz[0]
            if j in int_set and abs(coef - 1.0) < eps:
                bounded.add(j)
    return len(bounded) == len(int_set) and len(int_set) > 0


def _is_feasible(x, A, b, int_set, eps):
    n = len(x)
    if any(x[j] < -eps for j in range(n)):
        return False
    for j in int_set:
        if abs(x[j] - round(x[j])) > eps:
            return False
    for i, row in enumerate(A):
        lhs = sum(row[j] * x[j] for j in range(n))
        if lhs > b[i] + eps:
            return False
    return True


def _round_binary(lp_solution, int_set, c, A, b, minimize, eps):
    # Greedy rounding with local search improvement
    n = len(lp_solution)
    sol = list(lp_solution)
    sign = 1 if minimize else -1

    # Round fractional vars, preferring low-impact first
    candidates = [(sign * c[j], lp_solution[j], j) for j in int_set
                  if abs(lp_solution[j] - round(lp_solution[j])) > eps]
    candidates.sort()

    for _, val, j in candidates:
        rounded = round(val)
        sol[j] = float(rounded)

        feasible = True
        for i, row in enumerate(A):
            if sum(row[k] * sol[k] for k in range(n)) > b[i] + eps:
                feasible = False
                break

        if not feasible:
            sol[j] = 1.0 - rounded
            feasible = True
            for i, row in enumerate(A):
                if sum(row[k] * sol[k] for k in range(n)) > b[i] + eps:
                    feasible = False
                    break
            if not feasible:
                return None

    if not _is_feasible(sol, A, b, int_set, eps):
        return None

    # Phase 1: flip improvement
    improved = True
    while improved:
        improved = False
        flip_candidates = [(sign * c[j], j) for j in int_set
                          if (minimize and sol[j] > 0.5) or (not minimize and sol[j] < 0.5)]
        flip_candidates.sort()

        for _, j in flip_candidates:
            old_val = sol[j]
            sol[j] = 1.0 - old_val
            if _is_feasible(sol, A, b, int_set, eps):
                improved = True
            else:
                sol[j] = old_val

    # Phase 2: swap improvement
    improved = True
    while improved:
        improved = False
        zeros = [j for j in int_set if sol[j] < 0.5]
        ones = [j for j in int_set if sol[j] > 0.5]

        best_gain, best_swap = 0, None
        for j_on in zeros:
            gain_on = -sign * c[j_on]
            for j_off in ones:
                net_gain = gain_on + sign * c[j_off]
                if net_gain > best_gain:
                    sol[j_on], sol[j_off] = 1.0, 0.0
                    if _is_feasible(sol, A, b, int_set, eps):
                        best_gain, best_swap = net_gain, (j_on, j_off)
                    sol[j_on], sol[j_off] = 0.0, 1.0

        if best_swap:
            j_on, j_off = best_swap
            sol[j_on], sol[j_off] = 1.0, 0.0
            improved = True

    return tuple(sol)


def _lns_improve(solution, c, A, b, int_set, minimize, eps, max_iter, iterations, destroy_frac, rng):
    n = len(solution)
    int_list = list(int_set)
    k = max(1, int(len(int_list) * destroy_frac))

    def objective_fn(sol):
        return sum(c[j] * sol[j] for j in range(n))

    def destroy(sol, rng):
        unfixed = set(rng.sample(int_list, min(k, len(int_list))))
        return (sol, unfixed)

    def repair(partial, _):
        sol, unfixed = partial
        candidate = _solve_sub_mip(sol, c, A, b, int_set, unfixed, minimize, eps, max_iter)
        return candidate if candidate else sol

    result = _lns(
        solution, objective_fn, destroy, repair,
        minimize=minimize, max_iter=iterations, max_no_improve=iterations,
        seed=rng.randint(0, 2**31),
    )
    return result.solution, result.evaluations


def _solve_sub_mip(current_sol, c, A, b, int_set, free_vars, minimize, eps, max_iter):
    n = len(c)
    sign = 1 if minimize else -1

    lower = [0.0] * n
    upper = [float("inf")] * n
    for j in int_set:
        if j in free_vars:
            lower[j], upper[j] = 0.0, 1.0
        else:
            lower[j] = upper[j] = current_sol[j]

    result = _solve_node(c, A, b, lower, upper, minimize, eps, max_iter)
    if result.status != LPStatus.OPTIMAL:
        return None

    sol = list(result.solution)
    frac_vars = [j for j in free_vars if abs(sol[j] - round(sol[j])) > eps]
    if not frac_vars:
        return tuple(sol)

    # Small B&B for sub-problem
    best_sol, best_obj = None, float("inf") if minimize else float("-inf")

    rounded = list(sol)
    for j in free_vars:
        rounded[j] = round(rounded[j])
    if _is_feasible(rounded, A, b, int_set, eps):
        best_sol = tuple(rounded)
        best_obj = sum(c[j] * rounded[j] for j in range(n))

    stack = [(list(lower), list(upper))]
    nodes = 0

    while stack and nodes < 100:
        lo, hi = stack.pop()
        nodes += 1

        res = _solve_node(c, A, b, lo, hi, minimize, eps, max_iter)
        if res.status != LPStatus.OPTIMAL:
            continue
        if best_sol is not None and sign * res.objective >= sign * best_obj - eps:
            continue

        branch_var, best_frac = None, 0
        for j in free_vars:
            frac = abs(res.solution[j] - round(res.solution[j]))
            if frac > eps and frac > best_frac:
                branch_var, best_frac = j, frac

        if branch_var is None:
            obj = res.objective
            if sign * obj < sign * best_obj:
                best_sol, best_obj = tuple(res.solution), obj
            continue

        val = res.solution[branch_var]
        lo_down, hi_down = list(lo), list(hi)
        hi_down[branch_var] = floor(val)
        stack.append((lo_down, hi_down))

        lo_up, hi_up = list(lo), list(hi)
        lo_up[branch_var] = ceil(val)
        stack.append((lo_up, hi_up))

    return best_sol
