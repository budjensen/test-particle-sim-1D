"""
particles.py

Defines the Species class: a Structure-of-Arrays (SoA) container that stores
properties of particles belonging to one species (e.g., electrons or ions).

"""

# import statements
from __future__ import annotations
import numpy as np

# species class definition
class Species:
    """
    A simple Structure-of-Arrays (SoA) representation of a particle species.

    Each physical quantity (position, velocity, etc.) is stored in its own
    contiguous NumPy array for efficient vectorized operations.

    Attributes
    ----------
    q : float
        charge of one particle (Coulombs, C)
    m : float
        mass of one particle (kilograms, kg)
    name : str
        species name, "electron", "ion", etc.
    dtype : np.dtype
        floating-point precision for numeric arrays
    capacity : int
        allocated length of all arrays (number of available slots)
    N : int
        current number of active particles
    x, vx : np.ndarray
        arrays for position (m) and velocity (m/s) (1D simulation)
    weight : np.ndarray
        particle weights (used for scaling particle contribution to charge density)
    alive : np.ndarray
        boolean array marking whether each particle is active
    """

    # constructor
    def __init__(
        self,
        q: float,
        m: float,
        name: str = "electron",
        capacity: int = 0,
        dtype=np.float64,
    ) -> None:
        """Initialize empty species container"""
        # store physical constants and identifiers
        self.q = float(q)   # float
        self.m = float(m)   # float
        self.name = name    # string
        self.dtype = dtype  # floating point type

        self.capacity = int(capacity)
        self.N = 0  # number of active particles

        # allocate SoA storage (1D arrays for 1D sim); preallocate if capacity > 0
        self.x  = np.zeros(self.capacity, dtype=self.dtype)
        self.vx = np.zeros(self.capacity, dtype=self.dtype)

        # allocate auxiliary bookkeeping arrays
        self.weight = np.ones(self.capacity, dtype=self.dtype) # weighting factors
        self.alive  = np.ones(self.capacity, dtype=bool)       # true = active

    # public methods
    def add_particles(self, x_init: np.ndarray, vx_init: np.ndarray) -> None:
        """
        Append new particles to this species

        Parameters
        ----------
        x_init : np.ndarray
            Initial positions of new particles
        vx_init : np.ndarray
            Initial velocities of new particles

        Notes
        -----
        function assumes both arrays are the same length
        arrays are appended to the end of the current data
        """

        # number of new particles being added
        n_new = len(x_init)

        # check if enough space; if not, allocate more
        self._ensure_capacity(self.N + n_new)

        # compute slice indices for where to put new data
        start = self.N
        end = self.N + n_new

        # copy incoming values into array slices
        # np.asarray ensures consistent dtype
        self.x[start:end] = np.asarray(x_init, dtype=self.dtype)
        self.vx[start:end] = np.asarray(vx_init, dtype=self.dtype)

        # initialize other arrays for new particles
        self.weight[start:end] = 1.0   # default weight
        self.alive[start:end] = True   # mark as active

        # update particle count
        self.N = end

    def initialize_uniform(
        self,
        n: int,
        x_min: float,
        x_max: float,
        v0: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """
        Create N particles uniformly distributed in space

        Parameters
        ----------
        n : int
            number of particles to create
        x_min, x_max : float
            spatial bounds for uniform distribution
        v0 : float, optional
            initial velocity assigned to all particles (default 0)
        seed : int or None, optional
            random seed for reproducibility
        """
        # initialize random number generator (NumPy’s modern API)
        rng = np.random.default_rng(seed)

        # generate uniformly spaced random positions
        x_new = rng.uniform(x_min, x_max, size=n).astype(self.dtype)

        # all velocities start with the same value v0
        vx_new = np.full(n, v0, dtype=self.dtype)

        # reuse add_particles to avoid repeating logic (DRY principle)
        self.add_particles(x_new, vx_new)

    def __len__(self) -> int:
        return self.N

    # internal helpers
    def _ensure_capacity(self, needed: int) -> None:
        """ Expand storage arrays if needed """

        # guard pattern: early return if no expansion needed
        if needed <= self.capacity:
            return

        # compute new capacity
        new_cap = max(needed, max(1, int(self.capacity * 2)))

        # grow all SoA arrays to the same new size
        self.x = self._grow(self.x, new_cap)
        self.vx = self._grow(self.vx, new_cap)
        self.weight = self._grow(self.weight, new_cap, fill=1.0)
        self.alive = self._grow(self.alive, new_cap, fill=True, array_dtype=bool)

        # update capacity
        self.capacity = new_cap

    @staticmethod
    def _grow(
        arr: np.ndarray,
        new_cap: int,
        fill=0,
        array_dtype=None,
    ) -> np.ndarray:
        """
        Create a larger copy of existing array

        Parameters
        ----------
        arr : np.ndarray
            old array to copy from
        new_cap : int
            desired total length
        fill : scalar
            value to fill the new (empty) portion with
        array_dtype : np.dtype or None
            optionally override dtype (used for bool arrays)

        Returns
        -------
        np.ndarray
            new array containing the old data plus new filler elements
        """
        # pick the right data type
        dtype = array_dtype or arr.dtype

        # allocate a new array of desired size
        out = np.empty(new_cap, dtype=dtype)

        # copy existing values into the front section
        n_old = len(arr)
        out[:n_old] = arr

        # fill the remainder with a sensible default value
        if dtype == bool:
            out[n_old:] = bool(fill)
        else:
            out[n_old:] = fill

        return out
