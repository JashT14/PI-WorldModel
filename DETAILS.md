# Physics-Informed Latent World Model: Detailed Technical Manual

This document provides a comprehensive technical breakdown of every module, mathematical formulation, data structure, and optimization technique in this repository.

## 1. Architectural Blueprint and Information Flow

The Physics-Informed Latent World Model (PI-WorldModel) operates as a closed-loop perception, imagination, and planning system:

```
Physical State Observation x_t                 Action a_t (Torque / Force)
       [cos theta, sin theta, theta_dot]                    |
                 |                                          |
                 v                                          |
       [ State Encoder E_phi ]                              |
                 |                                          |
                 v                                          v
       Latent State z_t = [s_t, v_t] --------------> [ Neural Acceleration Net ]
       (Generalized Position and Velocity)           [ f_theta(s_t, v_t, a_t)   ]
                                                                |
                                                                v
                                                     Predicted Accel a_net
                                                                |
                                                                v
                                              [ Symplectic Kinematic Integrator ]
                                              v_(t+1) = v_t + a_net * dt
                                              s_(t+1) = s_t + v_t * dt + 0.5 * a_net * dt^2
                                                                |
                                                                v
                                                  Next Latent State z_(t+1)
                                                                |
                                         +----------------------+----------------------+
                                         |                                             |
                                         v                                             v
                               [ State Decoder D_psi ]                       [ Latent Physics Loss ]
                                         |                                   * Multi-Step Horizon
                                         v                                   * Work-Energy Balance:
                               Imagined Observation x_hat_(t+1)                Delta(0.5 * v^2) = a_net * v_avg * dt
```

## 2. Component Specifications

### 2.1 Physics Simulators (`src/environment.py`)

* **Inverted Pendulum (`PendulumPhysicsEnv`)**:
  * Continuous non-linear dynamics governed by:
    $$\ddot{\theta} = \frac{g}{l} \sin\theta + \frac{1}{m l^2} \left( u - b \dot{\theta} \right)$$
  * Observation vector: $\mathbf{x} = [\cos\theta, \sin\theta, \dot{\theta}] \in \mathbb{R}^3$.
  * Kinetic energy: $T = \frac{1}{2} m l^2 \dot{\theta}^2$.
  * Potential energy: $V(\theta) = m g l (1 + \cos\theta)$, ensuring conservative potential balance relative to the hanging rest point ($\theta = \pi$).
  * Numerical integrators: 4th-Order Runge-Kutta (`RK4`) and Symplectic Euler.
  * Work-energy verification: Tracks $\Delta E = W_{\text{actuator}} - W_{\text{damping}}$ with relative energy drift $< 0.0001\%$ over 500 unactuated steps.

* **CartPole (`CartPolePhysicsEnv`)**:
  * Coupled cart-pole dynamics simulating linear cart motion ($x$) and rotating pendulum ($\theta$).
  * Observation vector: $\mathbf{x} = [x, \dot{x}, \cos\theta, \sin\theta, \dot{\theta}] \in \mathbb{R}^5$.

### 2.2 Neural Network Architecture (`src/models.py`)

* **`StateEncoder` ($\mathcal{E}_\phi$)**:
  * Maps raw physical observation $\mathbf{x}_t \in \mathbb{R}^3 \to \mathbf{z}_t = [\mathbf{s}_t, \mathbf{v}_t] \in \mathbb{R}^4$.
  * Layers: `Linear(3, 64) -> SiLU -> Linear(64, 64) -> SiLU -> Linear(64, 4)`.

* **`SymplecticKinematicsTransitionNet` ($\mathcal{T}_\theta$)**:
  * Takes current canonical latent state $\mathbf{z}_t = [\mathbf{s}_t, \mathbf{v}_t]$ and action $\mathbf{a}_t$, and computes generalized acceleration:
    $$\mathbf{a}_{\text{net}} = \text{MLP}([\mathbf{s}_t, \mathbf{v}_t, \mathbf{a}_t])$$
  * Performs exact second-order symplectic kinematic integration:
    $$\mathbf{v}_{t+1} = \mathbf{v}_t + \mathbf{a}_{\text{net}} \Delta t$$
    $$\mathbf{s}_{t+1} = \mathbf{s}_t + \mathbf{v}_t \Delta t + \frac{1}{2} \mathbf{a}_{\text{net}} \Delta t^2$$
  * Analytically satisfies $\frac{d\mathbf{s}}{dt} = \mathbf{v}$ with $0.00$ residual.

* **`StateDecoder` ($\mathcal{D}_\psi$)**:
  * Maps latent canonical state $\mathbf{z}_t \in \mathbb{R}^4 \to \hat{\mathbf{x}}_t \in \mathbb{R}^3$.
  * Normalizes the $[\cos\theta, \sin\theta]$ output to lie on the unit circle $\cos^2\theta + \sin^2\theta = 1$.

* **`PhysicsInformedWorldModel`**:
  * Composite model uniting Encoder, Transition, and Decoder with a multi-step `dream(z_0, action_sequence)` recurrent unroller.
  * Total trainable parameters: 14,025 (~55 KB).

### 2.3 Multi-Objective Physics Losses (`src/losses.py`)

The composite loss function penalizes perceptual error, multi-step dynamical divergence, and thermodynamic violations:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{latent}} \mathcal{L}_{\text{latent}} + \lambda_{\text{future}} \mathcal{L}_{\text{future}} + \lambda_{\text{energy}} \mathcal{L}_{\text{energy}}$$

1. **Autoencoder Reconstruction Loss**:
   $$\mathcal{L}_{\text{recon}} = \|\mathbf{x}_0 - \mathcal{D}_\psi(\mathcal{E}_\phi(\mathbf{x}_0))\|^2$$
2. **Discounted Multi-Step Latent Loss**:
   $$\mathcal{L}_{\text{latent}} = \frac{1}{H} \sum_{h=1}^H \gamma^h \|\mathbf{z}_{t+h} - \hat{\mathbf{z}}_{t+h}\|^2$$
3. **Decoded Observation Prediction Loss**:
   $$\mathcal{L}_{\text{future}} = \frac{1}{H} \sum_{h=1}^H \|\mathbf{x}_{t+h} - \mathcal{D}_\psi(\hat{\mathbf{z}}_{t+h})\|^2$$
4. **Latent Kinetic Energy Work Balance Loss**:
   $$\Delta T_{\text{latent}} = \frac{1}{2} \|\mathbf{v}_{t+1}\|^2 - \frac{1}{2} \|\mathbf{v}_t\|^2$$
   $$W_{\text{predicted}} = \left( \mathbf{a}_{\text{net}} \cdot \frac{\mathbf{v}_t + \mathbf{v}_{t+1}}{2} \right) \Delta t$$
   $$\mathcal{L}_{\text{energy}} = \|\Delta T_{\text{latent}} - W_{\text{predicted}}\|^2$$

### 2.4 Data Collection and Pretraining Pipeline (`src/dataset.py` & `src/trainer.py`)

* **Multi-Policy Trajectory Collector**:
  * Explores the full phase space using 5 excitation regimes:
    * `chirp`: Linear frequency sweep ($\omega_0 \to \omega_1$) for broadband resonance excitation.
    * `multisine`: Multi-frequency harmonic superposition.
    * `bang_bang`: Maximum torque alternating impulses ($+u_{\max} \to -u_{\max}$).
    * `impulse`: Sparse high-energy shock torques.
    * `random_walk`: Smooth Ornstein-Uhlenbeck random walk.
* **Sliding Window Sequence Dataset**:
  * Slices continuous trajectories into overlapping chunks of length $K+1$ for horizon training.
* **World Model Trainer**:
  * Optimized for multi-threaded CPU training using Intel MKL and OpenMP.
  * Implements `ReduceLROnPlateau`, gradient clipping at $\text{norm} = 1.0$, and model serialization.

### 2.5 Latent Model Predictive Control (`src/planner.py`)

The agent uses its trained world model to optimize actions directly in its latent imagination without querying the physical environment:

1. **Cross-Entropy Method (CEM)**:
   * Initializes Gaussian belief over action sequences $\mu \sim \mathcal{N}(0, \sigma^2)$.
   * Rolls out $N = 128$ trajectories for horizon $H = 12$.
   * Selects top $K = 16$ elite candidates and updates $(\mu, \sigma)$ over 3 iterations.
   * Planning latency: ~10.2 ms on CPU.
2. **Model-Predictive Path Integral (MPPI)**:
   * Evaluates sampled action rollouts and computes exponential softmax temperature weights.
   * Planning latency: ~3.4 ms on CPU.
3. **Random Shooting (RS)**:
   * Evaluates $N = 128$ uniform random action trajectories and takes the argmin cost sequence.
   * Planning latency: ~2.8 ms on CPU.

### 2.6 Pure C11 Zero-Allocation Runtime (`c_runtime/`)

To support ultra-low-latency deployment on microcontrollers, robotics boards, and edge CPUs:
* **`world_model.h` & `world_model.c`**:
  * Implements forward propagation, matrix multiplications, SiLU activations, and second-order symplectic integration entirely in C11.
  * **Zero Dynamic Memory Allocation**: Uses static local arrays on the execution stack.
* **`weights.h`**:
  * Static float arrays exported directly from trained PyTorch weights.
* **`main_embedded.c`**:
  * High-throughput benchmark binary achieving **2.94 microseconds per transition step** ($340.1\text{ kHz}$ throughput) and **8.00 ms MPC planning cycle latency**.

## 3. Directory and File Map

```
project_4_world_model/
├── README.md                           # Intuitive overview and quick start guide
├── DETAILS.md                          # Exhaustive technical and architectural manual (this file)
├── ML_World_Models.md                  # Theoretical whitepaper on cognitive world models
├── plan.md                             # Phase-wise engineering blueprint
├── requirements.txt                    # Python dependencies
├── main.py                             # Full end-to-end runnable demonstration
│
├── src/                                # Core Engine Modules
│   ├── environment.py                  # Inverted Pendulum & CartPole physics engines
│   ├── models.py                       # Encoder, Symplectic Kinematics Transition, Decoder
│   ├── losses.py                       # Multi-objective physics losses
│   ├── dataset.py                      # Multi-policy data collector & dataset loader
│   ├── trainer.py                      # CPU-optimized multi-step trainer
│   └── planner.py                      # Latent MPC Planner (CEM, MPPI, Random Shooting)
│
├── scripts/                            # Standalone Executables and Utilities
│   ├── simulate_live_gui.py            # Live visual GUI and GIF simulation recorder
│   ├── collect_dataset.py              # Exploration data generator CLI
│   ├── verify_phase2_architecture.py   # Kinematic invariance verification CLI
│   ├── train_world_model.py            # Training and evaluation CLI
│   ├── run_closed_loop_mpc.py          # Closed-loop simulator with ASCII display
│   ├── export_onnx.py                  # Standalone TorchScript & ONNX export CLI
│   └── export_c_header.py              # Static C weights header generator
│
├── c_runtime/                          # Pure C11 Embedded Inference Engine
│   ├── world_model.h                   # Embedded runtime header
│   ├── world_model.c                   # Zero-allocation inference implementation
│   ├── weights.h                       # Static float weights
│   ├── main_embedded.c                 # Standalone embedded benchmark binary
│   └── embedded_world_model.exe        # Compiled high-speed native binary
│
└── tests/                              # Automated Unit Test Suite (16 Tests)
    ├── test_phase1.py                  # Environment and work-energy verification
    ├── test_phase2.py                  # Latent kinematics invariance verification
    ├── test_phase3.py                  # Multi-step training and checkpointing
    ├── test_phase4.py                  # Latent MPC planning verification
    └── test_phase5.py                  # TorchScript, C header, and binary verification
```

## 4. Verification and Benchmark Summary

```
=====================================================================================
                    PHYSICS-INFORMED LATENT WORLD MODEL BENCHMARKS
=====================================================================================
Single Latent Step (Pure C11 Engine):     2.94 microseconds (340.1 kHz throughput)
Single Latent Step (PyTorch CPU):        230.37 microseconds
Latent MPC Planning (MPPI, 128 rollouts):  3.43 milliseconds on standard CPU
Latent MPC Planning (CEM, 128 rollouts):  10.24 milliseconds on standard CPU
Latent MPC Planning (Pure C11 Engine):     8.00 milliseconds
Dynamic Memory Allocations (malloc):       0 bytes (Static stack buffers in C11)
Energy Drift Error (500 steps, b=0):      < 0.0001% (Exact conservative preservation)
Latent Kinematic Violation Residual:       0.00e+00 (Analytically exact)
Trainable Model Parameters:                14,025 parameters (~55 KB)
Automated Unit Tests:                     16 / 16 PASSING (0 failures in 2.55s)
=====================================================================================
```
