"""
particles.py


Defines the ``Species`` class: a Structure-of-Arrays (SoA) container that stores
properties of particles belonging to one species (e.g., electrons or ions) in a
1D-in-space, 3D-in-velocity (1D3V) plasma simulation.


The class keeps positions, velocity components, weights, and alive flags in
separate NumPy arrays for efficient vectorized operations.


Classes
-------
Species
   Container for one particle species (e.g., electrons, ions).


Examples
--------
Create an electron species and populate it with uniformly distributed particles:


>>> import numpy as np
>>> from test_particle_sim_1d.particles import Species
>>> electrons = Species(q=-1.0, m=1.0, name="electron", capacity=100)
>>> electrons.initialize_uniform(n=10, z_min=-0.1, z_max=0.1, v0=0.0, seed=42)
>>> len(electrons)
10
>>> electrons.z.shape
(100,)
"""

from __future__ import annotations

import numpy as np


# species class definition
class Species:
    """
    Structure-of-Arrays (SoA) container for a single particle species.


    Each physical quantity (position, velocity components, etc.) is stored in
    its own contiguous NumPy array for efficient vectorized operations in a
    1D-in-space, 3D-in-velocity (1D3V) simulation.


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
    z, vx : np.ndarray
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
        """
        Initialize an empty species container.


        Parameters
        ----------
        q : float
            Charge of one particle in Coulombs.
        m : float
            Mass of one particle in kilograms.
        name : str, optional
            Label for this species, e.g., "electron" or "ion".
        capacity : int, optional
            Initial capacity (number of particle slots) to allocate.
        dtype : numpy.dtype, optional
            Floating-point dtype used for internal arrays (default numpy.float64).


        Returns
        -------
        None
        """

        # store physical constants and identifiers
        self.q = float(q)  # float
        self.m = float(m)  # float
        self.name = name  # string
        self.dtype = dtype  # floating point type

        self.capacity = int(capacity)
        self.N = 0  # number of active particles

        # allocate SoA storage (1D arrays for 1D sim); preallocate if capacity > 0
        self.z = np.zeros(self.capacity, dtype=self.dtype)
        self.vx = np.zeros(self.capacity, dtype=self.dtype)
        self.vy = np.zeros(self.capacity, dtype=self.dtype)
        self.vz = np.zeros(self.capacity, dtype=self.dtype)

        # allocate auxiliary bookkeeping arrays
        self.weight = np.ones(self.capacity, dtype=self.dtype)  # weighting factors
        self.alive = np.ones(self.capacity, dtype=bool)  # true = active

    # public methods
    def add_particles(self, z_init: np.ndarray, v_init: np.ndarray) -> None:
        """
        Append new particles to this species.


        Parameters
        ----------
        z_init : np.ndarray
            Initial positions of new particles, shape ``(N,)``.
        v_init : np.ndarray
            Initial velocities of new particles, shape ``(N, 3)``, ordered
            as ``(vx, vy, vz)``.


        Returns
        -------
        None


        Notes
        -----
        The two input arrays must have the same leading dimension ``N``.
        New particles are appended to the end of the existing data.
        """

        # number of new particles being added
        n_new = len(z_init)

        # check if enough space; if not, allocate more
        self._ensure_capacity(self.N + n_new)

        # compute slice indices for where to put new data
        start = self.N
        end = self.N + n_new

        # copy incoming values into array slices
        # np.asarray ensures consistent dtype
        self.z[start:end] = np.asarray(z_init, dtype=self.dtype)
        self.vx[start:end] = v_init[:, 0]
        self.vy[start:end] = v_init[:, 1]
        self.vz[start:end] = v_init[:, 2]

        # initialize other arrays for new particles
        self.weight[start:end] = 1.0  # default weight
        self.alive[start:end] = True  # mark as active

        # update particle count
        self.N = end

    def initialize_uniform(
        self,
        n: int,
        z_min: float,
        z_max: float,
        v0: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """
        Create particles uniformly distributed in space.


        Parameters
        ----------
        n : int
            Number of particles to create.
        z_min : float
            Lower bound of the position domain.
        z_max : float
            Upper bound of the position domain.
        v0 : float or array_like, optional
            Initial velocity assigned to all particles. If scalar, it is
            interpreted as a single component value; if array-like, it should
            broadcast to shape ``(3,)``. Default is 0.0.
        seed : int or None, optional
            Random seed for reproducibility.


        Returns
        -------
        None
        """
        # initialize random number generator (NumPy's modern API)
        rng = np.random.default_rng(seed)

        # generate uniformly spaced random positions
        z_new = rng.uniform(z_min, z_max, size=n).astype(self.dtype)

        # all velocities start with the same value v0
        v_new = (
            np.zeros((n, 3), dtype=self.dtype) if v0 is None else np.tile(v0, (n, 1))
        )

        # reuse add_particles to avoid repeating logic (DRY principle)
        self.add_particles(z_new, v_new)

    def __len__(self) -> int:
        """
        Number of particles currently stored in this species.


        Returns
        -------
        int
            The number of particles ``N``.
        """
        return self.N

    # internal helpers
    def _ensure_capacity(self, needed: int) -> None:
        """
        Ensure that internal arrays can hold at least ``needed`` particles.


        Parameters
        ----------
        needed : int
            Required number of particle slots.


        Returns
        -------
        None
        """

        # guard pattern: early return if no expansion needed
        if needed <= self.capacity:
            return

        # compute new capacity
        new_cap = max(needed, 1, int(self.capacity * 2))

        # grow all SoA arrays to the same new size
        self.z = self._grow(self.z, new_cap)
        self.vx = self._grow(self.vx, new_cap)
        self.vy = self._grow(self.vy, new_cap)
        self.vz = self._grow(self.vz, new_cap)
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
        Create a larger copy of an existing 1D array.


        Parameters
        ----------
        arr : np.ndarray
            Original array to copy from.
        new_cap : int
            Desired total length of the returned array.
        fill : scalar, optional
            Value used to initialize the newly allocated portion
            (elements from ``len(arr)`` to ``new_cap - 1``).
        array_dtype : numpy.dtype or None, optional
            Optional dtype override for the returned array. If None,
            ``arr.dtype`` is used.


        Returns
        -------
        np.ndarray
            New array of length ``new_cap`` containing the original data in
            the first ``len(arr)`` entries and the ``fill`` value in the rest.
        """

        # pick the right data type
        dtype = array_dtype or arr.dtype

        # allocate a new array of desired size
        out = np.empty(new_cap, dtype=dtype)

        # copy existing values into the front section
        n_old = len(arr)
        out[:n_old] = arr

        # fill the remainder with a sensible default value
        if dtype is bool:
            out[n_old:] = bool(fill)
        else:
            out[n_old:] = fill

        return out
