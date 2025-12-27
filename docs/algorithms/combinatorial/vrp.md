# solve_vrptw

Vehicle Routing Problem with Time Windows. Route vehicles to serve customers within capacity and time constraints.

## Example

```python
from solvor import Customer, solve_vrptw

# Customers at various locations (depot is separate)
customers = [
    Customer(id=1, x=2, y=3, demand=10, tw_start=0, tw_end=50, service_time=5),
    Customer(id=2, x=5, y=1, demand=15, tw_start=10, tw_end=60, service_time=5),
    Customer(id=3, x=3, y=4, demand=20, tw_start=20, tw_end=70, service_time=5),
]

result = solve_vrptw(
    customers,
    vehicles=2,
    depot=(0, 0),
    vehicle_capacity=30
)
print(result.solution.routes)  # Routes for each vehicle
print(result.objective)        # Total distance
```

## Customer

```python
@dataclass
class Customer:
    id: int                        # Unique customer ID
    x: float                       # X coordinate
    y: float                       # Y coordinate
    demand: float = 0.0            # Demand to serve
    tw_start: float = 0.0          # Earliest service time
    tw_end: float = inf            # Latest service time
    service_time: float = 0.0      # Service duration
    required_vehicles: int = 1     # For multi-resource visits
```

## The Problem

- Each vehicle starts and ends at depot
- Visit all customers within time windows
- Don't exceed vehicle capacity
- Minimize total distance

## See Also

- [Cookbook: TSP](../../cookbook/tsp.md) - Single vehicle, no constraints
- [Metaheuristics](../metaheuristics/index.md) - For larger instances
