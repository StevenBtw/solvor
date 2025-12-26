# particle_swarm

Swarm intelligence. Particles fly through the search space, attracted to their personal best and the global best. Like peer pressure for optimization.

## Example

```python
from solvor import particle_swarm

def objective(x):
    return sum(xi**2 for xi in x)

result = particle_swarm(
    objective,
    bounds=[(-10, 10)] * 5,
    n_particles=30,
    max_iter=1000
)
print(result.solution)  # Close to [0, 0, 0, 0, 0]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `bounds` | List of (min, max) for each dimension |
| `n_particles` | Swarm size (default 30) |
| `inertia` | Inertia weight/momentum (default 0.7) |
| `cognitive` | Personal best attraction (default 1.5) |
| `social` | Global best attraction (default 1.5) |
| `inertia_decay` | If set, linearly decay inertia to this value |
| `initial_positions` | Warm-start with known good positions |

## How It Works

1. Initialize particles with random positions and velocities
2. Evaluate all particles
3. Update personal and global bests
4. Update velocities: v = inertia·v + cognitive·r1·(pbest - x) + social·r2·(gbest - x)
5. Update positions: x = x + v
6. Repeat

## Tips

- **Velocity clamping built-in.** Particles won't yeet into infinity.
- **Good for exploration.** Swarm naturally spreads out.
- **Fewer parameters than GA.** Easier to tune.

## See Also

- [Differential Evolution](differential-evolution.md) - Another population method
