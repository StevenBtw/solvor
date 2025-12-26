# lns / alns

Large Neighborhood Search. Destroy part of your solution and rebuild it better. ALNS (Adaptive LNS) learns which operators work best.

## Signature

```python
def lns[T](
    initial: T,
    objective_fn: Callable[[T], float],
    destroy: Callable[[T, Random], T],
    repair: Callable[[T, Random], T],
    *,
    max_iter: int = 1000,
    max_no_improve: int = 100,
    accept: str = "improving",  # "improving", "accept_all", "simulated_annealing"
    on_progress: Callable[[Progress], bool | None] | None = None,
    progress_interval: int = 0,
) -> Result[T]

def alns[T](
    initial: T,
    objective_fn: Callable[[T], float],
    destroy_ops: Sequence[Callable[[T, Random], T]],
    repair_ops: Sequence[Callable[[T, Random], T]],
    *,
    max_iter: int = 10000,
    segment_size: int = 100,
    reaction_factor: float = 0.1,
    on_progress: Callable[[Progress], bool | None] | None = None,
    progress_interval: int = 0,
) -> Result[T]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `initial` | Starting solution |
| `objective_fn` | Function to minimize |
| `destroy` / `destroy_ops` | Destroy operator(s) taking (solution, Random) |
| `repair` / `repair_ops` | Repair operator(s) taking (partial, Random) |
| `max_iter` | Maximum iterations |
| `reaction_factor` | How fast ALNS adapts weights (0.1 = 10% per segment) |

## Example

```python
from solvor import lns, alns
from random import Random

# LNS: single destroy/repair operators (receive Random instance)
def destroy(solution, rng: Random):
    # Remove some elements
    solution = list(solution)
    idx = rng.randint(0, len(solution) - 1)
    solution[idx] = None
    return solution

def repair(partial, rng: Random):
    # Fill in missing elements
    return [0 if v is None else v for v in partial]

result = lns(initial, objective, destroy, repair, max_iter=1000)

# ALNS: multiple operators, learns which work best
def destroy_random(sol, rng: Random):
    # ... remove randomly
    return sol

def destroy_worst(sol, rng: Random):
    # ... remove highest cost elements
    return sol

def repair_greedy(partial, rng: Random):
    # ... insert greedily
    return partial

result = alns(initial, objective, [destroy_random, destroy_worst], [repair_greedy])
```

## How It Works

**LNS:**
1. Select random destroy and repair operators
2. Destroy part of solution
3. Repair to get new solution
4. Accept if better (or probabilistically)
5. Repeat

**ALNS adds:**
- Track success of each operator
- Increase weight for successful operators
- Select operators by weight

## Tips

- **Multiple destroy operators.** Random, worst-cost, cluster-based, etc.
- **Multiple repair operators.** Greedy, regret, random.
- **Destruction degree.** Destroy 10-40% of solution typically.

## See Also

- [tabu_search](tabu.md) - Simpler local search
- [Cookbook: Job Shop](../../cookbook/job-shop.md) - Scheduling example
