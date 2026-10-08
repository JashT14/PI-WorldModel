#ifndef WORLD_MODEL_H
#define WORLD_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#define WM_OBS_DIM 3
#define WM_ACTION_DIM 1
#define WM_LATENT_POS_DIM 2
#define WM_LATENT_VEL_DIM 2
#define WM_LATENT_DIM 4
#define WM_HIDDEN_DIM 64


//Encodes physical observation [cos(th), sin(th), th_dot] into canonical latent coordinates [s, v].

void wm_encode(const float obs[WM_OBS_DIM], float z_out[WM_LATENT_DIM]);


//Predicts acceleration a_net and integrates using exact symplectic Newtonian kinematics:
//v_(t+1) = v_t + a_net * dt
//s_(t+1) = s_t + v_t * dt + 0.5 * a_net * dt^2

void wm_transition_step(
    const float z_in[WM_LATENT_DIM],
    float action,
    float dt,
    float z_out[WM_LATENT_DIM],
    float a_net_out[WM_LATENT_POS_DIM]
);

//Decodes latent canonical state [s, v] back to observation space [cos, sin, vel].

void wm_decode(const float z_in[WM_LATENT_DIM], float obs_out[WM_OBS_DIM]);

//Autoregressively rolls out a sequence of actions in latent imagination.
void wm_dream(
    const float z_0[WM_LATENT_DIM],
    const float* actions,
    int horizon,
    float dt,
    float* z_traj_out
);

//Evaluates candidate action sequences in latent space and returns the optimal torque.
float wm_plan_mpc(
    const float current_obs[WM_OBS_DIM],
    int num_candidates,
    int horizon,
    float dt,
    float max_torque
);

#ifdef __cplusplus
}
#endif

#endif
