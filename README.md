# test-particle-sim-1D
Test particle simulator for APC 524

Many graduate-level PIC codes are not designed for computational efficiency. One main focus of this code will be writing a simple but high-performance Python-based test particle code that emphasizes clean software design and efficient data handling. 

The code will be structured around a Particle class that can initialize arrays of particles, calculate electric and magnetic fields, and integrate particle motion given time step, background density, and other simulation parameters. The main goals are to:
Implement a minimal framework that can later include collisions and magnetic fields.
Ensure the code is modular, efficient, and easy to expand.

Primary distinction between the two data storage strategies:

In an Array of Structures (AoS), each particle object contains its own properties—position, velocity, charge, and mass. These are stored together in memory for each particle. This layout is straightforward but inefficient for large-scale operations, since it requires looping through many small objects in memory.

In a Structure of Arrays (SoA), particle properties are stored as separate arrays (one array for all positions, one for all velocities, etc.). This allows operations to be vectorized and performed efficiently using NumPy without loops.

The Structure of Arrays approach will be implemented for better performance and cache locality, especially important as particle counts scale to tens or hundreds of thousands. This choice also simplifies vectorized updates in Python, where loop-based approaches are inherently slower.

The initial goal is to implement a collisionless test-particle simulation. The code will:

Initialize charged particles with specified masses, charges, and velocities.
Integrate their motion through predefined electric or magnetic fields.
Record diagnostics such as particle positions, velocities, and energies.

The first validation case will simulate a magnetic mirror, where a static magnetic field varies along one spatial dimension:
The field will be strongest at both ends of the domain and weakest at the center.
Particles initialized at the midplane should reflect at the appropriate turning points due to conservation of the adiabatic invariant (μB = constant).

This test case verifies correct motion integration and conservation laws without the need for a self-consistent field solve or collisions. The simulation will use an input file structure specifying:

Simulation domain and time step.
Field definitions (E, B, or both).
Particle information (mass, charge, count, and initial velocities/positions).
Collisions (to be added later).

The input system should easily switch between cases, such as electrons in a magnetic mirror or ions in a sheath.
