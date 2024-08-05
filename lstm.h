#ifndef LSTM_H
#define LSTM_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    int in_size;
    int hid_size;
    float *Wf, *Wi, *Wc, *Wo;
    float *Uf, *Ui, *Uc, *Uo;
    float *bf, *bi, *bc, *bo;
    float *h, *c;
} LSTMCell;

typedef struct {
    int num_cells;
    LSTMCell **cells;
} LSTMLayer;

typedef struct {
    float *W;
    float b;
} Out;

float sigmoid(float x);
float he_init(int fan_in);
float* alloc_init(size_t size, float (*init_func)(int), int fan_in);
LSTMCell* init_lstm_cell(int in_size, int hid_size);
LSTMLayer* init_lstm_layer(int num_cells, int in_size, int hid_size);
Out* init_out(int in_size);
void lstm_cell_forward(LSTMCell* cell, float* input);
void lstm_layer_forward(LSTMLayer* layer, float* input);
float out_forward(Out* out, float* input, int in_size);
void free_lstm_cell(LSTMCell* cell);
void free_lstm_layer(LSTMLayer* layer);
void free_out(Out* out);
void update_weights(float* weight, float grad, float lr);

#endif
