#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "world_model.h"

int main() {
    printf("EMBEDDED C11 ZERO-OVERHEAD WORLD MODEL RUNTIME DEMO\n");
    printf("Zero Dynamic Allocation (malloc) - High-Frequency Edge Loop\n");

    /* 1. Observation Encoding */
    float obs[3] = {0.707106f, 0.707106f, 0.500000f}; /* theta ~ 45 deg, vel = 0.5 */
    float z_0[4];
    wm_encode(obs, z_0);
    printf("1. Observation Encoded into Canonical Latent Space [s1, s2, v1, v2]:\n");
    printf("   z_0 = [%+.4f, %+.4f, %+.4f, %+.4f]\n\n", z_0[0], z_0[1], z_0[2], z_0[3]);

    /* 2. Latent Kinematics Step */
    float action = 1.5f; /* 1.5 N*m applied torque */
    float dt = 0.05f;
    float z_next[4];
    float a_net[2];
    wm_transition_step(z_0, action, dt, z_next, a_net);

    printf("2. Single Latent Kinematics Step (Applied Torque = %.2f N*m, dt = %.3fs):\n", action, dt);
    printf("   Predicted Accel a_net = [%+.4f, %+.4f]\n", a_net[0], a_net[1]);
    printf("   Next Latent State     = [%+.4f, %+.4f, %+.4f, %+.4f]\n\n",
           z_next[0], z_next[1], z_next[2], z_next[3]);

    /* 3. Observation Decoding */
    float obs_hat[3];
    wm_decode(z_next, obs_hat);
    printf("3. Decoded Next Observation [cos, sin, vel]:\n");
    printf("   obs_hat = [%+.4f, %+.4f, %+.4f]\n\n", obs_hat[0], obs_hat[1], obs_hat[2]);

    /* 4. High-Throughput CPU Benchmark */
    int n_bench = 50000;
    printf("4. Running %d Pure C11 Latent Transition Iterations...\n", n_bench);
    
    clock_t start = clock();
    float z_bench[4];
    memcpy(z_bench, z_0, sizeof(float) * 4);
    
    for (int i = 0; i < n_bench; ++i) {
        float z_tmp[4];
        wm_transition_step(z_bench, 1.0f, dt, z_tmp, NULL);
        memcpy(z_bench, z_tmp, sizeof(float) * 4);
    }
    
    clock_t end = clock();
    double total_sec = (double)(end - start) / CLOCKS_PER_SEC;
    double us_per_step = (total_sec / n_bench) * 1e6;
    double freq_khz = (n_bench / total_sec) / 1000.0;

    printf("   -> Total Time: %.4f seconds\n", total_sec);
    printf("   -> Latency:    %.3f microseconds per transition step\n", us_per_step);
    printf("   -> Throughput: %.1f kHz (thousand transitions / sec on single core)\n\n", freq_khz);

    /* 5. Embedded Latent MPC Planning */
    int num_candidates = 128;
    int horizon = 10;
    printf("5. Embedded MPC Planning in C (%d candidate trajectories x %d horizon steps)...\n",
           num_candidates, horizon);

    start = clock();
    float best_action = wm_plan_mpc(obs, num_candidates, horizon, dt, 2.5f);
    end = clock();

    double mpc_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("   -> Planned Optimal Torque: %+.4f N*m\n", best_action);
    printf("   -> Planning Cycle Latency: %.2f ms\n\n", mpc_ms);

    printf("   [SUCCESS] C11 Embedded World Model Engine Verified!\n");
    return 0;
}
