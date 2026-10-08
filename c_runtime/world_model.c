#include "world_model.h"
#include "weights.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Helper: Dense Layer Forward Pass (out = activation(W * in + b)) */
static void dense_layer(
    const float* in,
    const float* weights,
    const float* bias,
    int in_dim,
    int out_dim,
    float* out,
    int use_tanh
) {
    for (int i = 0; i < out_dim; ++i) {
        float sum = bias[i];
        const float* w_row = &weights[i * in_dim];
        for (int j = 0; j < in_dim; ++j) {
            sum += w_row[j] * in[j];
        }
        out[i] = use_tanh ? tanhf(sum) : sum;
    }
}

void wm_encode(const float obs[WM_OBS_DIM], float z_out[WM_LATENT_DIM]) {
    float h1[WM_HIDDEN_DIM];
    float h2[WM_HIDDEN_DIM];

    /* Layer 1: Linear(3, 64) + Tanh */
    dense_layer(obs, wm_encoder_net_0_weight, wm_encoder_net_0_bias, WM_OBS_DIM, WM_HIDDEN_DIM, h1, 1);
    /* Layer 2: Linear(64, 64) + Tanh */
    dense_layer(h1, wm_encoder_net_2_weight, wm_encoder_net_2_bias, WM_HIDDEN_DIM, WM_HIDDEN_DIM, h2, 1);
    /* Layer 3: Linear(64, 4) - Linear Output [s1, s2, v1, v2] */
    dense_layer(h2, wm_encoder_net_4_weight, wm_encoder_net_4_bias, WM_HIDDEN_DIM, WM_LATENT_DIM, z_out, 0);
}

void wm_transition_step(
    const float z_in[WM_LATENT_DIM],
    float action,
    float dt,
    float z_out[WM_LATENT_DIM],
    float a_net_out[WM_LATENT_POS_DIM]
) {
    /* Inputs: [s1, s2, v1, v2, action] (dim = 5) */
    float in_vec[5];
    in_vec[0] = z_in[0];
    in_vec[1] = z_in[1];
    in_vec[2] = z_in[2];
    in_vec[3] = z_in[3];
    in_vec[4] = action;

    float h1[WM_HIDDEN_DIM];
    float h2[WM_HIDDEN_DIM];
    float a_pred[WM_LATENT_POS_DIM];

    /* Layer 1: Linear(5, 64) + Tanh */
    dense_layer(in_vec, wm_transition_accel_net_0_weight, wm_transition_accel_net_0_bias, 5, WM_HIDDEN_DIM, h1, 1);
    /* Layer 2: Linear(64, 64) + Tanh */
    dense_layer(h1, wm_transition_accel_net_2_weight, wm_transition_accel_net_2_bias, WM_HIDDEN_DIM, WM_HIDDEN_DIM, h2, 1);
    /* Layer 3: Linear(64, 2) - Linear Output [a1, a2] */
    dense_layer(h2, wm_transition_accel_net_4_weight, wm_transition_accel_net_4_bias, WM_HIDDEN_DIM, WM_LATENT_POS_DIM, a_pred, 0);

    if (a_net_out != NULL) {
        a_net_out[0] = a_pred[0];
        a_net_out[1] = a_pred[1];
    }

    /* Exact Symplectic Newtonian Kinematic Integration:
       v_(t+1) = v_t + a_net * dt
       s_(t+1) = s_t + v_t * dt + 0.5 * a_net * dt^2 */
    float s0 = z_in[0];
    float s1 = z_in[1];
    float v0 = z_in[2];
    float v1 = z_in[3];

    float v0_next = v0 + a_pred[0] * dt;
    float v1_next = v1 + a_pred[1] * dt;

    float s0_next = s0 + v0 * dt + 0.5f * a_pred[0] * (dt * dt);
    float s1_next = s1 + v1 * dt + 0.5f * a_pred[1] * (dt * dt);

    z_out[0] = s0_next;
    z_out[1] = s1_next;
    z_out[2] = v0_next;
    z_out[3] = v1_next;
}

void wm_decode(const float z_in[WM_LATENT_DIM], float obs_out[WM_OBS_DIM]) {
    float h1[WM_HIDDEN_DIM];
    float h2[WM_HIDDEN_DIM];

    /* Layer 1: Linear(4, 64) + Tanh */
    dense_layer(z_in, wm_decoder_net_0_weight, wm_decoder_net_0_bias, WM_LATENT_DIM, WM_HIDDEN_DIM, h1, 1);
    /* Layer 2: Linear(64, 64) + Tanh */
    dense_layer(h1, wm_decoder_net_2_weight, wm_decoder_net_2_bias, WM_HIDDEN_DIM, WM_HIDDEN_DIM, h2, 1);
    /* Layer 3: Linear(64, 3) - Linear Output [cos_th, sin_th, th_dot] */
    dense_layer(h2, wm_decoder_net_4_weight, wm_decoder_net_4_bias, WM_HIDDEN_DIM, WM_OBS_DIM, obs_out, 0);
}

void wm_dream(
    const float z_0[WM_LATENT_DIM],
    const float* actions,
    int horizon,
    float dt,
    float* z_traj_out
) {
    float z_curr[WM_LATENT_DIM];
    memcpy(z_curr, z_0, sizeof(float) * WM_LATENT_DIM);
    memcpy(z_traj_out, z_curr, sizeof(float) * WM_LATENT_DIM);

    for (int t = 0; t < horizon; ++t) {
        float z_next[WM_LATENT_DIM];
        wm_transition_step(z_curr, actions[t], dt, z_next, NULL);
        memcpy(z_curr, z_next, sizeof(float) * WM_LATENT_DIM);
        memcpy(&z_traj_out[(t + 1) * WM_LATENT_DIM], z_curr, sizeof(float) * WM_LATENT_DIM);
    }
}

/* Fast Pseudo-Random Float Generator for Embedded MPC */
static float fast_rand_uniform(float min_val, float max_val) {
    float scale = (float)rand() / (float)RAND_MAX;
    return min_val + scale * (max_val - min_val);
}

float wm_plan_mpc(
    const float current_obs[WM_OBS_DIM],
    int num_candidates,
    int horizon,
    float dt,
    float max_torque
) {
    float z_0[WM_LATENT_DIM];
    wm_encode(current_obs, z_0);

    float best_cost = 1e30f;
    float best_action = 0.0f;

    for (int i = 0; i < num_candidates; ++i) {
        float z_curr[WM_LATENT_DIM];
        memcpy(z_curr, z_0, sizeof(float) * WM_LATENT_DIM);

        float candidate_first_action = 0.0f;
        float total_cost = 0.0f;

        for (int t = 0; t < horizon; ++t) {
            float action = fast_rand_uniform(-max_torque, max_torque);
            if (t == 0) {
                candidate_first_action = action;
            }

            float z_next[WM_LATENT_DIM];
            wm_transition_step(z_curr, action, dt, z_next, NULL);
            memcpy(z_curr, z_next, sizeof(float) * WM_LATENT_DIM);

            float obs_hat[WM_OBS_DIM];
            wm_decode(z_curr, obs_hat);

            float cos_th = obs_hat[0];
            float sin_th = obs_hat[1];
            float th_dot = obs_hat[2];

            /* Cost function: Upright (cos=1, sin=0, vel=0) */
            float angle_cost = (1.0f - cos_th) + 0.5f * (sin_th * sin_th);
            float vel_cost = 0.1f * (th_dot * th_dot);
            float act_cost = 0.005f * (action * action);

            total_cost += angle_cost + vel_cost + act_cost;
        }

        if (total_cost < best_cost) {
            best_cost = total_cost;
            best_action = candidate_first_action;
        }
    }

    return best_action;
}
