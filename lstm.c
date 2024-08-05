#include "lstm.h"

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float he_init(int fan_in) {
    return ((float)rand() / RAND_MAX) * sqrtf(2.0f / fan_in);
}

float* alloc_init(size_t size, float (*init_func)(int), int fan_in) {
    float* arr = malloc(size * sizeof(float));
    if (!arr) exit(1);

    if (init_func) {
        for (size_t i = 0; i < size; i++) {
            arr[i] = init_func(fan_in);
        }
    } else {
        memset(arr, 0, size * sizeof(float));
    }

    return arr;
}

LSTMCell* init_lstm_cell(int in_size, int hid_size) {
    LSTMCell* cell = malloc(sizeof(LSTMCell));
    if (!cell) exit(1);

    cell->in_size = in_size;
    cell->hid_size = hid_size;

    size_t in_hid_size = in_size * hid_size;
    size_t hid_hid_size = hid_size * hid_size;

    cell->Wf = alloc_init(in_hid_size, he_init, in_size);
    cell->Wi = alloc_init(in_hid_size, he_init, in_size);
    cell->Wc = alloc_init(in_hid_size, he_init, in_size);
    cell->Wo = alloc_init(in_hid_size, he_init, in_size);

    cell->Uf = alloc_init(hid_hid_size, he_init, hid_size);
    cell->Ui = alloc_init(hid_hid_size, he_init, hid_size);
    cell->Uc = alloc_init(hid_hid_size, he_init, hid_size);
    cell->Uo = alloc_init(hid_hid_size, he_init, hid_size);

    cell->bf = alloc_init(hid_size, NULL, 0);
    cell->bi = alloc_init(hid_size, NULL, 0);
    cell->bc = alloc_init(hid_size, NULL, 0);
    cell->bo = alloc_init(hid_size, NULL, 0);

    cell->h = alloc_init(hid_size, NULL, 0);
    cell->c = alloc_init(hid_size, NULL, 0);

    for (int i = 0; i < hid_size; i++) {
        cell->bf[i] = 1.0f;
    }

    return cell;
}

LSTMLayer* init_lstm_layer(int num_cells, int in_size, int hid_size) {
    LSTMLayer* layer = malloc(sizeof(LSTMLayer));
    if (!layer) exit(1);

    layer->num_cells = num_cells;
    layer->cells = malloc(num_cells * sizeof(LSTMCell*));
    if (!layer->cells) exit(1);

    for (int i = 0; i < num_cells; i++) {
        layer->cells[i] = init_lstm_cell(in_size, hid_size);
    }

    return layer;
}

Out* init_out(int in_size) {
    Out* out = malloc(sizeof(Out));
    if (!out) exit(1);

    out->W = alloc_init(in_size, he_init, in_size);
    out->b = 0.0f;

    return out;
}

void lstm_cell_forward(LSTMCell* cell, float* input) {
    float* f = alloc_init(cell->hid_size, NULL, 0);
    float* i = alloc_init(cell->hid_size, NULL, 0);
    float* c_tilde = alloc_init(cell->hid_size, NULL, 0);
    float* o = alloc_init(cell->hid_size, NULL, 0);
    float* c_next = alloc_init(cell->hid_size, NULL, 0);
    float* h_next = alloc_init(cell->hid_size, NULL, 0);

    void calc_gate(float* gate, float* W, float* U, float* b) {
        for (int j = 0; j < cell->hid_size; j++) {
            gate[j] = b[j];
            for (int k = 0; k < cell->in_size; k++) {
                gate[j] += W[j * cell->in_size + k] * input[k];
            }
            for (int k = 0; k < cell->hid_size; k++) {
                gate[j] += U[j * cell->hid_size + k] * cell->h[k];
            }
            gate[j] = sigmoid(gate[j]);
        }
    }

    calc_gate(f, cell->Wf, cell->Uf, cell->bf);
    calc_gate(i, cell->Wi, cell->Ui, cell->bi);
    calc_gate(o, cell->Wo, cell->Uo, cell->bo);

    for (int j = 0; j < cell->hid_size; j++) {
        c_tilde[j] = cell->bc[j];
        for (int k = 0; k < cell->in_size; k++) {
            c_tilde[j] += cell->Wc[j * cell->in_size + k] * input[k];
        }
        for (int k = 0; k < cell->hid_size; k++) {
            c_tilde[j] += cell->Uc[j * cell->hid_size + k] * cell->h[k];
        }
        c_tilde[j] = tanhf(c_tilde[j]);
    }

    for (int j = 0; j < cell->hid_size; j++) {
        c_next[j] = f[j] * cell->c[j] + i[j] * c_tilde[j];
        h_next[j] = o[j] * tanhf(c_next[j]);
    }

    memcpy(cell->c, c_next, cell->hid_size * sizeof(float));
    memcpy(cell->h, h_next, cell->hid_size * sizeof(float));

    free(f); free(i); free(c_tilde); free(o); free(c_next); free(h_next);
}

void lstm_layer_forward(LSTMLayer* layer, float* input) {
    for (int i = 0; i < layer->num_cells; i++) {
        lstm_cell_forward(layer->cells[i], input);
    }
}

float out_forward(Out* out, float* input, int in_size) {
    float result = out->b;
    for (int i = 0; i < in_size; i++) {
        result += out->W[i] * input[i];
    }
    return result;
}

void lstm_cell_backward(LSTMCell* cell, float* input, float* dh_next, float* dc_next, float* dx, float lr) {
    float* f = alloc_init(cell->hid_size, NULL, 0);
    float* i = alloc_init(cell->hid_size, NULL, 0);
    float* c_tilde = alloc_init(cell->hid_size, NULL, 0);
    float* o = alloc_init(cell->hid_size, NULL, 0);
    float* c_next = alloc_init(cell->hid_size, NULL, 0);
    float* h_next = alloc_init(cell->hid_size, NULL, 0);

    void calc_gate(float* gate, float* W, float* U, float* b) {
        for (int j = 0; j < cell->hid_size; j++) {
            gate[j] = b[j];
            for (int k = 0; k < cell->in_size; k++) {
                gate[j] += W[j * cell->in_size + k] * input[k];
            }
            for (int k = 0; k < cell->hid_size; k++) {
                gate[j] += U[j * cell->hid_size + k] * cell->h[k];
            }
            gate[j] = sigmoid(gate[j]);
        }
    }

    calc_gate(f, cell->Wf, cell->Uf, cell->bf);
    calc_gate(i, cell->Wi, cell->Ui, cell->bi);
    calc_gate(o, cell->Wo, cell->Uo, cell->bo);

    for (int j = 0; j < cell->hid_size; j++) {
        c_tilde[j] = cell->bc[j];
        for (int k = 0; k < cell->in_size; k++) {
            c_tilde[j] += cell->Wc[j * cell->in_size + k] * input[k];
        }
        for (int k = 0; k < cell->hid_size; k++) {
            c_tilde[j] += cell->Uc[j * cell->hid_size + k] * cell->h[k];
        }
        c_tilde[j] = tanhf(c_tilde[j]);
    }

    for (int j = 0; j < cell->hid_size; j++) {
        c_next[j] = f[j] * cell->c[j] + i[j] * c_tilde[j];
        h_next[j] = o[j] * tanhf(c_next[j]);
    }

    float* do_ = alloc_init(cell->hid_size, NULL, 0);
    float* dc_tilde = alloc_init(cell->hid_size, NULL, 0);
    float* di = alloc_init(cell->hid_size, NULL, 0);
    float* df = alloc_init(cell->hid_size, NULL, 0);

    for (int j = 0; j < cell->hid_size; j++) {
        do_[j] = dh_next[j] * tanhf(c_next[j]);
        dc_next[j] += dh_next[j] * o[j] * (1 - tanhf(c_next[j]) * tanhf(c_next[j]));
        dc_tilde[j] = dc_next[j] * i[j];
        di[j] = dc_next[j] * c_tilde[j];
        df[j] = dc_next[j] * cell->c[j];
    }

    void update_gate(float* dgate, float* gate, float* W, float* U, float* b) {
        for (int j = 0; j < cell->hid_size; j++) {
            float dgate_pre = dgate[j] * gate[j] * (1 - gate[j]);
            for (int k = 0; k < cell->in_size; k++) {
                dx[k] += dgate_pre * W[j * cell->in_size + k];
                update_weights(&W[j * cell->in_size + k], dgate_pre * input[k], lr);
            }
            for (int k = 0; k < cell->hid_size; k++) {
                dh_next[k] += dgate_pre * U[j * cell->hid_size + k];
                update_weights(&U[j * cell->hid_size + k], dgate_pre * cell->h[k], lr);
            }
            update_weights(&b[j], dgate_pre, lr);
        }
    }

    update_gate(df, f, cell->Wf, cell->Uf, cell->bf);
    update_gate(di, i, cell->Wi, cell->Ui, cell->bi);
    update_gate(do_, o, cell->Wo, cell->Uo, cell->bo);

    for (int j = 0; j < cell->hid_size; j++) {
        float dc_tilde_pre = dc_tilde[j] * (1 - c_tilde[j] * c_tilde[j]);
        for (int k = 0; k < cell->in_size; k++) {
            dx[k] += dc_tilde_pre * cell->Wc[j * cell->in_size + k];
            update_weights(&cell->Wc[j * cell->in_size + k], dc_tilde_pre * input[k], lr);
        }
        for (int k = 0; k < cell->hid_size; k++) {
            dh_next[k] += dc_tilde_pre * cell->Uc[j * cell->hid_size + k];
            update_weights(&cell->Uc[j * cell->hid_size + k], dc_tilde_pre * cell->h[k], lr);
        }
        update_weights(&cell->bc[j], dc_tilde_pre, lr);
    }

    free(f); free(i); free(c_tilde); free(o); free(c_next); free(h_next);
    free(do_); free(dc_tilde); free(di); free(df);
}

void lstm_layer_backward(LSTMLayer* layer, float* input, float* dh_next, float* dc_next, float* dx, float lr) {
    float* dh_prev = alloc_init(layer->cells[0]->hid_size, NULL, 0);
    float* dc_prev = alloc_init(layer->cells[0]->hid_size, NULL, 0);

    for (int i = layer->num_cells - 1; i >= 0; i--) {
        memcpy(dh_prev, dh_next, layer->cells[i]->hid_size * sizeof(float));
        memcpy(dc_prev, dc_next, layer->cells[i]->hid_size * sizeof(float));
        memset(dh_next, 0, layer->cells[i]->hid_size * sizeof(float));
        memset(dc_next, 0, layer->cells[i]->hid_size * sizeof(float));

        lstm_cell_backward(layer->cells[i], input, dh_prev, dc_prev, dx, lr);
    }

    free(dh_prev);
    free(dc_prev);
}

void free_lstm_cell(LSTMCell* cell) {
    free(cell->Wf); free(cell->Wi); free(cell->Wc); free(cell->Wo);
    free(cell->Uf); free(cell->Ui); free(cell->Uc); free(cell->Uo);
    free(cell->bf); free(cell->bi); free(cell->bc); free(cell->bo);
    free(cell->h); free(cell->c);
    free(cell);
}

void free_lstm_layer(LSTMLayer* layer) {
    for (int i = 0; i < layer->num_cells; i++) {
        free_lstm_cell(layer->cells[i]);
    }
    free(layer->cells);
    free(layer);
}

void free_out(Out* out) {
    free(out->W);
    free(out);
}

void update_weights(float* weight, float grad, float lr) {
    *weight -= lr * grad;
}
