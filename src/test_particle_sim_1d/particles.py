"""
particles.py


Defines the ``Species`` class: a Structure-of-Arrays (SoA) container that stores
properties of particles belonging to one species (e.g., electrons or ions) in a
1D-in-space, 3D-in-velocity (1D3V) plasma simulation.


The class keeps positions, velocity components, weights, and alive flags in
separate NumPy arrays for efficient vectorized operations.


Functions
---------
sample_maxwellian : Generate velocities from a 3D Maxwellian distribution.


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

from .initialization import constants


def sample_maxwellian(
    n: int,
    mass: float,
    temperature: dict[str, float],
    mean_velocity: float = 0.0,
    seed: int | None = None,
    dtype=np.float64,
) -> np.ndarray:
    """
    Generate velocities from a 3D Maxwellian (Gaussian) distribution.

    Parameters
    ----------
    n : int
        Number of samples to draw
    mass : float
        Particle mass [kg]
    temperature : dict[str, float]
        Particle temperature in Kelvin [K] or electronvolts [eV]
    mean_velocity : float, optional
        Mean drift velocity [m/s], default 0.0
    seed : int or None, optional
        RNG seed, default None
    dtype : np.dtype
        Floating-point type for output

    Returns
    -------
    np.ndarray
        Shape (n, 3) array of velocity components [vx, vy, vz]

    Notes
    -----
    Since a Maxwellian distribution describes velocity magnitudes,
    individual components are drawn from a Gaussian distribution with
    standard deviation `sigma = sqrt(kB * T / m)`.

    Examples
    --------
    >>> from test_particle_sim_1d import particles
    >>> from test_particle_sim_1d.initialization import constants
    >>> particles.sample_maxwellian(5, constants.m_e, {"eV": 10})
    array([[-1666438.13197273, -1851032.27214499,  -682821.19994243],
           [  439642.61815002,   335299.21954749,   956643.41448306],
           [ 1587744.48058107, -3142078.12313282,     3247.0552664 ],
           [ -366354.74190221, -2063735.35368675, -1601787.56530326],
           [  941907.82864779, -3161259.01940836, -1160794.96240072]])
    >>> particles.sample_maxwellian(5, constants.m_p, {"K": 300})
    array([[ 2531.56061941,  -462.15468457,  2552.48093799],
           [-2205.48337309,  -808.75778961,   933.82890129],
           [ 1328.18252258,  -285.73035999,  -777.86873709],
           [  858.85731176, -1099.90661809,  1598.40588797],
           [   43.15730554, -1070.52229508,  1076.59602439]])

    Raises
    ------
    ValueError
        If temperature dictionary does not contain 'K' or 'eV' key.
    """
    rng = np.random.default_rng(seed)

    if "eV" in temperature:
        sigma = np.sqrt(constants.q_e * temperature["eV"] / mass)
    elif "K" in temperature:
        sigma = np.sqrt(constants.kb * temperature["K"] / mass)
    else:
        error_msg = "Temperature dictionary must contain 'K' or 'eV' key."
        raise ValueError(error_msg)

    # Return 3D Gaussian-distributed velocities
    return rng.normal(loc=mean_velocity, scale=sigma, size=(n, 3)).astype(dtype)


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

    def remove_particles(self, indices: np.ndarray | int) -> None:
        """
        Remove particles at the specified indices.

        Parameters
        ----------
        indices : np.ndarray | int
            Indices of particles to remove.
        """
        # Copy the later particles to the positions of the removed ones
        if isinstance(indices, int):
            indices = np.array([indices])
        for idx in sorted(indices):
            # Move the last particle to the position of the removed one
            self._copy(self.N - 1, idx)
            self.N -= 1

    def _copy(self, old_idx: int, new_idx: int):
        """
        Move a particle from old_idx to new_idx. This does not
        change the total number of particles.
        """
        self.z[new_idx] = self.z[old_idx]
        self.vx[new_idx] = self.vx[old_idx]
        self.vy[new_idx] = self.vy[old_idx]
        self.vz[new_idx] = self.vz[old_idx]
        self.weight[new_idx] = self.weight[old_idx]
        self.alive[new_idx] = self.alive[old_idx]

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

    def initialize_maxwellian(
        self,
        n: int,
        z_min: float,
        z_max: float,
        temperature: dict[str, float],
        mean_velocity: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """
        Initialize particles uniformly in space with Maxwellian velocity distribution.

        Parameters
        ----------
        n : int
            Number of particles to initialize
        z_min, z_max : float
            Spatial bounds for uniform initialization
        temperature : dict[str, float]
            Particle temperature in Kelvin [K] or electronvolts [eV]
        mean_velocity : float, optional
            Mean drift velocity [m/s], default 0.0
        seed : int or None, optional
            RNG seed, default None

        Notes
        -----
        This method is designed to be called immediately after instance
        declaration. If particle positions or velocities are non-zero, a
        RuntimeError will be thrown to prevent overwriting data.

        Examples
        --------
        >>> from test_particle_sim_1d.particles import Species
        >>> from test_particle_sim_1d.initialization import constants
        >>> elec = Species(q=-constants.q_e, m=constants.m_e, name="electron", capacity=10)
        >>> elec.vx
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        >>> elec.initialize_maxwellian(n=10, z_min=0.0, z_max=1.0, temperature={"eV": 1.5})
        >>> elec.vx
        array([ -33033.04963604, -104607.77384604, -230482.34375015,
                496914.378235  , -820260.54488735,  185363.40818585,
                 -5937.95752475,  424445.8731878 ,  437843.29246543,
               -115207.57202853])

        Raises
        ------
        RuntimeError
            If particle arrays already contain data (non-zero entries)
        """
        # Check if any of z, vx, vy, vz have non-zero entries
        if (
            np.any(self.z[: self.N])
            or np.any(self.vx[: self.N])
            or np.any(self.vy[: self.N])
            or np.any(self.vz[: self.N])
        ):
            error_msg = "Cannot initialize Maxwellian velocities: particle arrays already contain data."
            raise RuntimeError(error_msg)
        rng = np.random.default_rng(seed)
        z_new = rng.uniform(z_min, z_max, size=n).astype(self.dtype)
        v_new = sample_maxwellian(
            n, self.m, temperature, mean_velocity, seed, dtype=self.dtype
        )
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
