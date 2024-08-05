#include "lstm.h"
#include "sine_wave.h"
#include <time.h>

int main() {
    srand(time(NULL));

    int in_size = 1;
    int hid_size = 10;
    int num_layers = 1;
    int seq_len = 1000;
    int train_iters = 30000;
    float lr = 0.00003f;

    LSTMLayer** layers = malloc(num_layers * sizeof(LSTMLayer*));
    if (!layers) exit(1);
    layers[0] = init_lstm_layer(1, in_size, hid_size);
    for (int i = 1; i < num_layers; i++) {
        layers[i] = init_lstm_layer(1, hid_size, hid_size);
    }

    Out* out = init_out(hid_size);

    float input, target;

    for (int iter = 0; iter < train_iters; iter++) {
        float total_loss = 0.0f;

        for (int l = 0; l < num_layers; l++) {
            memset(layers[l]->cells[0]->h, 0, hid_size * sizeof(float));
            memset(layers[l]->cells[0]->c, 0, hid_size * sizeof(float));
        }

        for (int t = 0; t < seq_len - 1; t++) {
            input = gen_sine_wave(t);
            target = gen_sine_wave(t + 1);

            lstm_layer_forward(layers[0], &input);
            for (int l = 1; l < num_layers; l++) {
                lstm_layer_forward(layers[l], layers[l-1]->cells[0]->h);
            }

            float prediction = out_forward(out, layers[num_layers-1]->cells[0]->h, hid_size);
            float loss = (prediction - target) * (prediction - target);
            total_loss += loss;

            float out_grad = 2 * (prediction - target);
            float* dh_next = calloc(hid_size, sizeof(float));
            float* dc_next = calloc(hid_size, sizeof(float));
            float* dx = calloc(in_size, sizeof(float));

            for (int j = 0; j < hid_size; j++) {
                dh_next[j] = out_grad * out->W[j];
            }

            for (int l = num_layers - 1; l >= 0; l--) {
                float* layer_input = (l == 0) ? &input : layers[l-1]->cells[0]->h;
                lstm_layer_backward(layers[l], layer_input, dh_next, dc_next, dx, lr);
            }

            for (int j = 0; j < hid_size; j++) {
                update_weights(&out->W[j], out_grad * layers[num_layers-1]->cells[0]->h[j], lr);
            }
            update_weights(&out->b, out_grad, lr);

            free(dh_next);
            free(dc_next);
            free(dx);
        }

        if (iter % 500 == 0) {
            printf("Iter %d, Avg Loss: %f\n", iter, total_loss / seq_len);
        }
    }

    printf("\nTesting the trained model:\n");
    FILE *y_true_file = fopen("y_true.txt", "w");
    FILE *y_pred_file = fopen("y_pred.txt", "w");
    if (!y_true_file || !y_pred_file) {
        printf("Error opening files for writing\n");
        exit(1);
    }

    for (int t = 0; t < 5000; t++) {
        input = gen_sine_wave(t);
        lstm_layer_forward(layers[0], &input);
        for (int l = 1; l < num_layers; l++) {
            lstm_layer_forward(layers[l], layers[l-1]->cells[0]->h);
        }
        target = gen_sine_wave(t + 1);

        float prediction = out_forward(out, layers[num_layers-1]->cells[0]->h, hid_size);
        fprintf(y_true_file, "%f\n", target);
        fprintf(y_pred_file, "%f\n", prediction);

        if (t % 10 == 0) {
            printf("Time %d: In = %.4f, Pred = %.4f, Next = %.4f\n",
                   t, input, prediction, target);
        }
    }

    fclose(y_true_file);
    fclose(y_pred_file);

    for (int l = 0; l < num_layers; l++) {
        free_lstm_layer(layers[l]);
    }
    free(layers);
    free_out(out);

    return 0;
}
