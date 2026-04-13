/**
 * MIT License
 * Copyright (c) [2026] The Barfing Fox [Andrea Nardinocchi
 * (andrea@nardinan.it)]
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* NARDINAN chapter:
 * basics math needed to implement all the mechanisms and the computation
 * between matrices and vectors.
 */
#define d_assert(c)                                                                                                                                            \
  do {                                                                                                                                                         \
    if (!(c)) {                                                                                                                                                \
      fprintf(stderr, "ABORT: condition '%s' failed (function %s, line %d)\n", #c, __FUNCTION__, __LINE__);                                                    \
      abort();                                                                                                                                                 \
    }                                                                                                                                                          \
  } while (0)
typedef float t_vector;
typedef float t_matrix;
struct s_matrix_header {
  size_t columns, rows;
};
float f_vector_dot_product(t_vector *vector_a, t_vector *vector_b, size_t length) {
  float result = 0;
  for (size_t index = 0; index < length; ++index)
    result += (vector_a[index] * vector_b[index]);
  return result;
}
void f_vector_softmax(t_vector *vector, size_t length) {
  float maximum_value, sum_exponentials = 0;
  maximum_value = vector[0];
  for (size_t index = 1; index < length; ++index)
    if (vector[index] > maximum_value)
      maximum_value = vector[index];
  for (size_t index = 0; index < length; ++index)
    sum_exponentials += expf(vector[index] - maximum_value);
  for (size_t index = 0; index < length; ++index)
    vector[index] = expf((vector[index] - maximum_value)) / sum_exponentials;
}
void f_vector_log_softmax(t_vector *vector, size_t length) {
  float maximum_value, sum_exponentials = 0;
  maximum_value = vector[0];
  for (size_t index = 1; index < length; ++index)
    if (vector[index] > maximum_value)
      maximum_value = vector[index];
  for (size_t index = 0; index < length; ++index)
    sum_exponentials += expf(vector[index] - maximum_value);
  for (size_t index = 0; index < length; ++index)
    vector[index] = vector[index] - maximum_value - logf(sum_exponentials);
}
/* matrices are stored as a linear vector of rows * columns, represented as a
 * rows * sequence of columns. This means: access to (C, R) = (0, 0) is element
 * (0) in the vector access to (C, R) = (1, 0) is element (1) in the vector
 * access to (C, R) = (2, 1) is element (columns * R) + (C)
 */
t_matrix *f_matrix_new(size_t rows, size_t columns) {
  struct s_matrix_header *result;
  if ((result = (struct s_matrix_header *) malloc(sizeof(struct s_matrix_header) + (sizeof(float) * rows * columns)))) {
    result->rows = rows;
    result->columns = columns;
    memset((t_matrix *) ((void *) result + sizeof(struct s_matrix_header)), 0, sizeof(float) * rows * columns);
  }
  return ((result) ? (t_matrix *) ((void *) result + sizeof(struct s_matrix_header)) : NULL);
}
size_t f_matrix_columns(t_matrix *matrix) {
  return ((struct s_matrix_header *) ((void *) matrix - sizeof(struct s_matrix_header)))->columns;
}
size_t f_matrix_rows(t_matrix *matrix) {
  return ((struct s_matrix_header *) ((void *) matrix - sizeof(struct s_matrix_header)))->rows;
}
t_matrix *f_matrix_duplicate(t_matrix *matrix) {
  struct s_matrix_header *result;
  size_t size_payload = (sizeof(struct s_matrix_header) + (sizeof(float) * f_matrix_rows(matrix) * f_matrix_columns(matrix)));
  if ((result = (struct s_matrix_header *) malloc(size_payload)))
    memcpy(result, ((void *) matrix - sizeof(struct s_matrix_header)), size_payload);
  return ((result) ? (t_matrix *) ((void *) result + sizeof(struct s_matrix_header)) : NULL);
}
t_matrix *f_matrix_new_arguments(size_t rows, size_t columns, ...) {
  t_matrix *result = f_matrix_new(rows, columns);
  if (result) {
    va_list arguments;
    va_start(arguments, columns);
    {
      for (size_t index = 0; index < (rows * columns); ++index)
        result[index] = va_arg(arguments, double); /* as before, va_arg promotes floats to double */
    }
    va_end(arguments);
  }
  return result;
}
#define d_matrix_getCR(m, c, r) (m[(f_matrix_columns(m) * (r)) + (c)])
#define d_matrix_getRC(m, r, c) (m[(f_matrix_columns(m) * (r)) + (c)])
void f_matrix_write(FILE *stream, t_matrix *matrix) {
  if (matrix) {
    fprintf(stream, "%zu %zu ", f_matrix_rows(matrix), f_matrix_columns(matrix));
    for (size_t index_row = 0; index_row < f_matrix_rows(matrix); ++index_row)
      for (size_t index_column = 0; index_column < f_matrix_columns(matrix); ++index_column)
        fprintf(stream, "%.9g ", d_matrix_getCR(matrix, index_column, index_row));
  }
}
t_matrix *f_matrix_read(FILE *stream, t_matrix *host) {
  t_matrix *result = NULL;
  size_t rows = 0, columns = 0;
  fscanf(stream, "%zu %zu ", &rows, &columns);
  if ((rows > 0) && (columns > 0)) {
    if (host) {
      d_assert((f_matrix_rows(host) == rows) && (f_matrix_columns(host) == columns));
      result = host;
    } else {
      result = f_matrix_new(rows, columns);
    }
    if (result)
      for (size_t index_row = 0; index_row < rows; ++index_row)
        for (size_t index_column = 0; index_column < columns; ++index_column) {
          float value;
          fscanf(stream, "%f ", &value);
          d_matrix_getCR(result, index_column, index_row) = value;
        }
  }
  return result;
}
void f_matrix_zero(t_matrix *matrix) {
  memset(matrix, 0, (f_matrix_rows(matrix) * f_matrix_columns(matrix)) * sizeof(float));
}
void f_matrix_free(t_matrix *matrix) {
  if (matrix)
    free((void *) matrix - (sizeof(struct s_matrix_header)));
}
t_matrix *f_matrix_multiply(t_matrix *host, t_matrix *matrix_a, t_matrix *matrix_b) {
  t_matrix *result = NULL;
  size_t vector_length;
  d_assert((vector_length = f_matrix_columns(matrix_a)) == f_matrix_rows(matrix_b));
  if (host) {
    d_assert((f_matrix_rows(host) == f_matrix_rows(matrix_a)) && (f_matrix_columns(host) == f_matrix_columns(matrix_b)));
    result = host;
  } else
    result = f_matrix_new(f_matrix_rows(matrix_a), f_matrix_columns(matrix_b));
  if (result)
    for (size_t index_row = 0; index_row < f_matrix_rows(result); ++index_row)
      for (size_t index_column = 0; index_column < f_matrix_columns(result); ++index_column) {
        d_matrix_getRC(result, index_row, index_column) = 0;
        for (size_t index_element_dot_product = 0; index_element_dot_product < vector_length; ++index_element_dot_product)
          d_matrix_getRC(result, index_row, index_column) += (d_matrix_getRC(matrix_a, index_row, index_element_dot_product) *
              d_matrix_getRC(matrix_b, index_element_dot_product, index_column));
      }
  return result;
}
void f_matrix_add_in_place(t_matrix *matrix_a, t_matrix *matrix_b) {
  d_assert((f_matrix_rows(matrix_a) == f_matrix_rows(matrix_b)) && (f_matrix_columns(matrix_a) == f_matrix_columns(matrix_b)));
  for (size_t index_row = 0; index_row < f_matrix_rows(matrix_a); ++index_row)
    for (size_t index_column = 0; index_column < f_matrix_columns(matrix_b); ++index_column)
      d_matrix_getRC(matrix_a, index_row, index_column) += d_matrix_getRC(matrix_b, index_row, index_column);
}
t_matrix *f_matrix_multiply_matrix_with_transposed(t_matrix *host, t_matrix *matrix_a, t_matrix *transposed_matrix_b) {
  t_matrix *result = NULL;
  size_t vector_length;
  d_assert((vector_length = f_matrix_columns(matrix_a)) == f_matrix_columns(transposed_matrix_b));
  if (host) {
    d_assert((f_matrix_rows(host) == f_matrix_rows(matrix_a)) && (f_matrix_columns(host) == f_matrix_rows(transposed_matrix_b)));
    result = host;
  } else
    result = f_matrix_new(f_matrix_rows(matrix_a), f_matrix_rows(transposed_matrix_b));
  if (result)
    for (size_t index_row = 0; index_row < f_matrix_rows(result); ++index_row)
      for (size_t index_column = 0; index_column < f_matrix_columns(result); ++index_column) {
        d_matrix_getRC(result, index_row, index_column) = 0;
        for (size_t index_element_dot_product = 0; index_element_dot_product < vector_length; ++index_element_dot_product)
          d_matrix_getRC(result, index_row, index_column) += (d_matrix_getRC(matrix_a, index_row, index_element_dot_product) *
              d_matrix_getCR(transposed_matrix_b, index_element_dot_product, index_column));
      }
  return result;
}
void f_matrix_multiply_matrix_with_transposed_add_into_host(t_matrix *host, t_matrix *matrix_a, t_matrix *transposed_matrix_b) {
  t_matrix *result = host;
  size_t vector_length;
  d_assert((vector_length = f_matrix_columns(matrix_a)) == f_matrix_columns(transposed_matrix_b) && (host) &&
      (f_matrix_rows(host) == f_matrix_rows(matrix_a)) && (f_matrix_columns(host) == f_matrix_rows(transposed_matrix_b)));
  if (result)
    for (size_t index_row = 0; index_row < f_matrix_rows(result); ++index_row)
      for (size_t index_column = 0; index_column < f_matrix_columns(result); ++index_column)
        for (size_t index_element_dot_product = 0; index_element_dot_product < vector_length; ++index_element_dot_product)
          d_matrix_getRC(result, index_row, index_column) += (d_matrix_getRC(matrix_a, index_row, index_element_dot_product) *
              d_matrix_getCR(transposed_matrix_b, index_element_dot_product, index_column));
}
t_matrix *f_matrix_multiply_transposed_with_matrix(t_matrix *host, t_matrix *transposed_matrix_a, t_matrix *matrix_b) {
  t_matrix *result = NULL;
  size_t vector_length;
  d_assert((vector_length = f_matrix_rows(transposed_matrix_a)) == f_matrix_rows(matrix_b));
  if (host) {
    d_assert((f_matrix_rows(host) == f_matrix_columns(transposed_matrix_a)) && (f_matrix_columns(host) == f_matrix_columns(matrix_b)));
    result = host;
  } else
    result = f_matrix_new(f_matrix_columns(transposed_matrix_a), f_matrix_columns(matrix_b));
  if (result)
    for (size_t index_row = 0; index_row < f_matrix_rows(result); ++index_row)
      for (size_t index_column = 0; index_column < f_matrix_columns(result); ++index_column) {
        d_matrix_getRC(result, index_row, index_column) = 0;
        for (size_t index_element_dot_product = 0; index_element_dot_product < vector_length; ++index_element_dot_product)
          d_matrix_getRC(result, index_row, index_column) += (d_matrix_getCR(transposed_matrix_a, index_row, index_element_dot_product) *
              d_matrix_getRC(matrix_b, index_element_dot_product, index_column));
      }
  return result;
}
void f_matrix_multiply_transposed_with_matrix_add_into_host(t_matrix *host, t_matrix *transposed_matrix_a, t_matrix *matrix_b) {
  t_matrix *result = host;
  size_t vector_length;
  d_assert((vector_length = f_matrix_rows(transposed_matrix_a)) == f_matrix_rows(matrix_b) && (host) &&
      (f_matrix_rows(host) == f_matrix_columns(transposed_matrix_a)) && (f_matrix_columns(host) == f_matrix_columns(matrix_b)));
  if (result)
    for (size_t index_row = 0; index_row < f_matrix_rows(result); ++index_row)
      for (size_t index_column = 0; index_column < f_matrix_columns(result); ++index_column)
        for (size_t index_element_dot_product = 0; index_element_dot_product < vector_length; ++index_element_dot_product)
          d_matrix_getRC(result, index_row, index_column) += (d_matrix_getCR(transposed_matrix_a, index_row, index_element_dot_product) *
              d_matrix_getRC(matrix_b, index_element_dot_product, index_column));
}
/* NARDINAN chapter:
 * encoding/decoding, we're now defining our vocabulary constraints. We'll set
 * some "special" numbers that are outside the printable ASCII table to encode
 * some special behaviors
 */
#define d_vocabulary_size 101
#define d_token_pad 0
#define d_token_bos 1 /* beginning of a sequence */
#define d_token_eos 2 /* end of sequence */
#define d_token_sys 3 /* system block (beginning) */
#define d_token_usr 4 /* user block (beginning) */
#define d_token_ast 5 /* assistant block (beginning) */
#define d_token_offset 6
size_t f_encode(const char *text, int *output_tokens, size_t output_length) {
  size_t length_text = strlen(text), index_token = 0;
  /* we are going to shift the printable characters (> 32) to have a sequence of
   * valid tokens that goes from 0 (pad) to 100 ('~' - 32 + 6) and compact
   * everything */
  for (size_t index_text = 0; (index_text < length_text) && (index_token < output_length); ++index_text) {
    if (isprint(text[index_text])) {
      output_tokens[index_token] = (text[index_text] - 32 /* first printable character */) + d_token_offset;
      ++index_token;
    }
  }
  return index_token;
}
void f_decode(const int *tokens, size_t tokens_length, char *output_text, size_t output_length) {
  size_t index_text = 0;
  for (size_t index_token = 0; (index_token < tokens_length) && (index_text < (output_length - 1)); ++index_token)
    if (tokens[index_token] >= d_token_offset) {
      output_text[index_text] = (tokens[index_token] - d_token_offset) + 32;
      ++index_text;
    }
  if (index_text < output_length)
    output_text[index_text] = 0;
}
/* d_model HAS TO BE even, otherwise f_positional_encoding_table_new() is going
 * to crash (or no, which is even worse) */
#define d_model 64 /* our amazing d_model size. I've seen that GPT2 small uses 768 but it will make something impossible to train with this crappy code */
#define d_context 512 /* maximum sequence of tokens we can use to guess the next token */
bool m_random_initialized = false;
float f_random_float(void) {
  /* well, I'll be using the Box-Muller transform, that should give a roughly
   * [-1, 1] normal distribution (more or less) as recommended */
  if (!m_random_initialized) {
    srand(time(NULL));
    m_random_initialized = true;
  }
  float unit_1 = ((rand() + 1.0f) / ((float) RAND_MAX + 1.0f)), unit_2 = ((rand() + 1.0f) / ((float) RAND_MAX + 1.0f));
  return sqrtf(-2.0f * logf(unit_1)) * cosf((2.0f * 3.14159265f) * unit_2) * 0.02f;
}
/* we need two matrices. The first one is the matrix where each row-vector
 * represents a token. The row N represents the token N. At the very beginning,
 * they're just random version that we'll adjust through backpropagation ...
 */
t_matrix *f_embedding_table_new(void) {
  t_matrix *result = f_matrix_new(d_vocabulary_size, d_model); /* this is going to be our embedding table
                                                                  for the vocabulary we've defined */
  if (result)
    for (size_t index_table = 0; index_table < (d_vocabulary_size * d_model); ++index_table)
      result[index_table] = f_random_float();
  return result;
}
float *f_embedding_table_get_token(t_matrix *embedding_table, int token_ID) {
  d_assert((embedding_table) && (token_ID >= 0) && (f_matrix_rows(embedding_table) > token_ID));
  return (float *) (&(d_matrix_getCR(embedding_table, 0, token_ID)));
}
/* we also need another matrix that tells us the position of the token in our
 * contexts, otherwise the word 'g''o''l''f' has the same weight of the word
 * 'o''g''f''l'. I'm following the specifications of the 'sinusoidal encoding'
 * paper, where the content of the encoding table is fixed. */
t_matrix *f_positional_encoding_table_new(size_t context_length) {
  t_matrix *result = NULL;
  d_assert((d_model % 2) == 0);
  if ((result = f_matrix_new(context_length, d_model)))
    for (size_t index_row = 0; index_row < context_length; ++index_row)
      for (size_t index_column = 0; index_column < d_model; index_column += 2) {
        float frequency = 1.0f / powf(10000.0f, ((float) index_column / (float) d_model));
        d_matrix_getCR(result, index_column, index_row) = sinf(index_row * frequency);
        d_matrix_getCR(result, (index_column + 1), index_row) = cosf(index_row * frequency);
      }
  return result;
}
float *f_positional_encoding_table_get_position(t_matrix *positional_encoding_table, size_t position) {
  d_assert((positional_encoding_table) && (f_matrix_rows(positional_encoding_table) > position));
  return (float *) (&(d_matrix_getCR(positional_encoding_table, 0, position)));
}
t_matrix *f_embedded_sequence_new(t_matrix *embedding_table, t_matrix *positional_encoding_table, const int *tokens, size_t tokens_length) {
  /* now we know that embedding_table's row N matches with the token at position
   * N, while positional_encoding_table's row N matches with the position of the
   * context */
  t_matrix *result = f_matrix_new(tokens_length, d_model);
  if (result)
    for (size_t index_sequence = 0; index_sequence < tokens_length; ++index_sequence) {
      /* we need to sum the elements of the embedding table with the vector of
       * positional encoding and store it into the result */
      float *token_row = f_embedding_table_get_token(embedding_table, tokens[index_sequence]),
            *position_row = f_positional_encoding_table_get_position(positional_encoding_table, index_sequence);
      for (size_t index_column = 0; index_column < d_model; ++index_column) {
        d_matrix_getCR(result, index_column, index_sequence) = (token_row[index_column] + position_row[index_column]);
      }
    }
  return result;
}
#define d_number_heads 4
/* we need a place where to store the intermediate results computed during the
 * feedforward process, to be sure that during the backpropagation we can obtain
 * those intermediates */
typedef struct s_transformer_internal_states {
  t_matrix *before_attention_normalized, *query[d_number_heads], *key[d_number_heads], *value[d_number_heads], *scores[d_number_heads],
      *attention_weights[d_number_heads], *attention_output, *before_feedforward_normalized, *feedforward_hidden_layer_pre_GELU,
      *feedforward_hidden_layer_post_GELU, *after_attention_normalized, *output;
} s_transformer_internal_states;
void f_transformer_internal_states_free(s_transformer_internal_states *internal_states) {
  f_matrix_free(internal_states->before_attention_normalized);
  for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
    f_matrix_free(internal_states->query[index_head]);
    f_matrix_free(internal_states->key[index_head]);
    f_matrix_free(internal_states->value[index_head]);
    f_matrix_free(internal_states->scores[index_head]);
    f_matrix_free(internal_states->attention_weights[index_head]);
  }
  f_matrix_free(internal_states->attention_output);
  f_matrix_free(internal_states->before_feedforward_normalized);
  f_matrix_free(internal_states->feedforward_hidden_layer_pre_GELU);
  f_matrix_free(internal_states->feedforward_hidden_layer_post_GELU);
  f_matrix_free(internal_states->after_attention_normalized);
  f_matrix_free(internal_states->output);
  memset(internal_states, 0, sizeof(s_transformer_internal_states));
}

/* d_model HAS TO BE divisible by the d_number_heads, otherwise you'll screw up
 * the math in the generation of the encoded attention weights */
#define d_size_head (d_model / d_number_heads)
#define d_W 0 /* weights */
#define d_G 1 /* gradient */
/* our attention is given by a d_number_head number of matrices that are going
 * to be used to tailor different kind of attentions that each token in the
 * sequence gives to another token. For instance, if I use the sentence "the fox
 * jumps in the VIGNETO", and I take the token 'jumps', a kind of attention
 * might be "OK but, jumps where?" -> in the vigneto. Another kind of attention
 * might be "whom is jumping?" -> the fox. And so on. The final matrix
 * (weight_output) tells to the model how to merge together these different kind
 * of attentions.
 */
typedef struct s_attention_weights {
  t_matrix *weight_query[d_number_heads][2], *weight_key[d_number_heads][2], *weight_value[d_number_heads][2], *weight_output[2];
} s_attention_weights;
void f_attention_weights_free(s_attention_weights *attention_weights) {
  if (attention_weights) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
        f_matrix_free(attention_weights->weight_query[index_head][index_matrix]);
        f_matrix_free(attention_weights->weight_key[index_head][index_matrix]);
        f_matrix_free(attention_weights->weight_value[index_head][index_matrix]);
      }
      f_matrix_free(attention_weights->weight_output[index_matrix]);
    }
    free(attention_weights);
  }
}
/* this structure contains the weights matrices (d_model x d_head) that drive
 * our attention mechanism. In particular we have the query (what a token is
 * looking for), the key (what it offers to others), and value (what it shares,
 * when picked). Based on this we're going to get another matrix that is going
 * to represent
 * - for each token - how's related to the other tokens. The bigger d_head is,
 * the more details we can identify during the back propagation, improving the
 * quality of the bounds between tokens. At the beginning - as usual -
 * everything is completely random */
s_attention_weights *f_attention_weights_new(void) {
  s_attention_weights *result = (s_attention_weights *) malloc(sizeof(s_attention_weights));
  d_assert((d_model % d_number_heads) == 0);
  for (size_t index_head = 0; (result) && (index_head < d_number_heads); ++index_head) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      result->weight_query[index_head][index_matrix] = f_matrix_new(d_model, d_size_head);
      result->weight_key[index_head][index_matrix] = f_matrix_new(d_model, d_size_head);
      result->weight_value[index_head][index_matrix] = f_matrix_new(d_model, d_size_head);
    }
    if ((result->weight_query[index_head][d_W]) && (result->weight_query[index_head][d_G]) && (result->weight_key[index_head][d_W]) &&
        (result->weight_key[index_head][d_G]) && (result->weight_value[index_head][d_W]) && (result->weight_value[index_head][d_G])) {
      for (size_t index_row = 0; index_row < d_model; ++index_row)
        for (size_t index_column = 0; index_column < d_size_head; ++index_column) {
          d_matrix_getCR(result->weight_query[index_head][d_W], index_column, index_row) = f_random_float();
          d_matrix_getCR(result->weight_key[index_head][d_W], index_column, index_row) = f_random_float();
          d_matrix_getCR(result->weight_value[index_head][d_W], index_column, index_row) = f_random_float();
        }
    } else {
      f_attention_weights_free(result);
      result = NULL;
    }
  }
  if (result) {
    if ((result->weight_output[d_W] = f_matrix_new(d_model, d_model)) && (result->weight_output[d_G] = f_matrix_new(d_model, d_model))) {
      for (size_t index_row = 0; index_row < d_model; ++index_row)
        for (size_t index_column = 0; index_column < d_model; ++index_column)
          d_matrix_getCR(result->weight_output[d_W], index_column, index_row) = f_random_float();
    } else {
      f_attention_weights_free(result);
      result = NULL;
    }
  }
  return result;
}
void f_attention_weights_gradient_zero(s_attention_weights *weights) {
  for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
    f_matrix_zero(weights->weight_query[index_head][d_G]);
    f_matrix_zero(weights->weight_key[index_head][d_G]);
    f_matrix_zero(weights->weight_value[index_head][d_G]);
  }
  f_matrix_zero(weights->weight_output[d_G]);
}
t_matrix *f_encoded_attention_weights_forward_new(s_transformer_internal_states *optional_internal_states, s_attention_weights *weights,
    t_matrix *embedded_sequence) {
  /* we start, firstly, by multiplying the weights matrices with the embedded
   * sequence. The embedded sequence is [sequence_length x d_model], while the
   * weight are [d_model x d_head] which means the the outcome should be
   * [sequence_length x (d_number_head x d_size_head)] */
  t_matrix *result = NULL, *collection_attention_kinds = f_matrix_new(f_matrix_rows(embedded_sequence), (d_number_heads * d_size_head)),
           *sequence_tokens_query = NULL, *sequence_tokens_key = NULL, *sequence_tokens_value = NULL, *scores = NULL, *scoped_attention_weights = NULL;
  for (size_t index_head = 0; (collection_attention_kinds) && (index_head < d_number_heads); ++index_head) {
    if (optional_internal_states) {
      if (optional_internal_states->query[index_head])
        f_matrix_free(optional_internal_states->query[index_head]);
      if (optional_internal_states->key[index_head])
        f_matrix_free(optional_internal_states->key[index_head]);
      if (optional_internal_states->value[index_head])
        f_matrix_free(optional_internal_states->value[index_head]);
      sequence_tokens_query = optional_internal_states->query[index_head] = f_matrix_multiply(NULL, embedded_sequence, weights->weight_query[index_head][d_W]);
      sequence_tokens_key = optional_internal_states->key[index_head] = f_matrix_multiply(NULL, embedded_sequence, weights->weight_key[index_head][d_W]);
      sequence_tokens_value = optional_internal_states->value[index_head] = f_matrix_multiply(NULL, embedded_sequence, weights->weight_value[index_head][d_W]);
    } else {
      sequence_tokens_query = f_matrix_multiply(sequence_tokens_query, embedded_sequence, weights->weight_query[index_head][d_W]);
      sequence_tokens_key = f_matrix_multiply(sequence_tokens_key, embedded_sequence, weights->weight_key[index_head][d_W]);
      sequence_tokens_value = f_matrix_multiply(sequence_tokens_value, embedded_sequence, weights->weight_value[index_head][d_W]);
    }
    if ((sequence_tokens_query) && (sequence_tokens_key) && (sequence_tokens_value)) {
      /* now that we know, for each token in the sequence its values for query,
       * key and value, we can compute the attention scores. The output matrix
       * is going to tell us for each [i][j] how much token i wants to attent to
       * token j: a large positive number it means A LOT, while near zero almost
       * nothing. To do so, I'm multiplying Q for K(transposed) a Q is
       * [sequence_length x d_head] and K(transposed) is [d_head x
       * seqence_length], the scores matrix is going to be [sequence_length x
       * sequence_length]
       */
      if (optional_internal_states) {
        if (optional_internal_states->scores[index_head])
          f_matrix_free(optional_internal_states->scores[index_head]);
        scores = optional_internal_states->scores[index_head] = f_matrix_multiply_matrix_with_transposed(NULL, sequence_tokens_query, sequence_tokens_key);
      } else {
        scores = f_matrix_multiply_matrix_with_transposed(scores, sequence_tokens_query, sequence_tokens_key);
      }
      if (scores) {
        /* as the scores value is the product of a lot of multiplications of
         * matrices, we might want to reduce the magnitude dividing everything
         * by sqrt of d_head: this is going to keep the value meaningful by
         * reducing its size
         */
        const float sqrt_head = sqrtf((float) d_size_head);
        for (size_t index_row = 0; index_row < f_matrix_rows(scores); ++index_row) {
          for (size_t index_column = 0; index_column < f_matrix_columns(scores); ++index_column) {
            /* additionally, we want to be sure that all the tokens that follow
             * a token at a specific position, are filtered out. If the row i
             * refers to the token at the position i of the embedded sequence,
             * in the row i all the entries from column (i + 1) are masked out.
             * We do this by setting their value to a very huge negative number
             * (e.g. -1e9)
             */
            if (index_column > index_row)
              d_matrix_getCR(scores, index_column, index_row) = -1e9;
            else
              d_matrix_getCR(scores, index_column, index_row) = (d_matrix_getCR(scores, index_column, index_row) / sqrt_head);
          }
          /* and for each row, I'm going to apply softmax */
          f_vector_softmax((t_vector *) &(d_matrix_getCR(scores, 0, index_row)), f_matrix_columns(scores));
        }
        /* and now, the final step! A new matrix with the final attention weight
         * is going to be generated, composed by the weights (our scores)
         * multiplied by the sequence_token_value. While the weights is a
         * [sequence_length x sequence_length] matrix and the
         * sequence_token_value is a [sequence_length x d_size_head] matrix, we
         * got a final matrix of [sequence_length x d_size_head].
         */
        if (optional_internal_states) {
          if (optional_internal_states->attention_weights[index_head])
            f_matrix_free(optional_internal_states->attention_weights[index_head]);
          scoped_attention_weights = optional_internal_states->attention_weights[index_head] = f_matrix_multiply(NULL, scores, sequence_tokens_value);
        } else {
          scoped_attention_weights = f_matrix_multiply(scoped_attention_weights, scores, sequence_tokens_value);
        }
        if (scoped_attention_weights)
          for (size_t index_row = 0; index_row < f_matrix_rows(embedded_sequence); ++index_row)
            for (size_t index_source_column = 0, index_destination_column = (d_size_head * index_head); index_source_column < d_size_head;
                ++index_source_column, ++index_destination_column) {
              d_matrix_getCR(collection_attention_kinds, index_destination_column, index_row) = d_matrix_getCR(scoped_attention_weights, index_source_column,
                  index_row);
            }
      }
    }
  }
  if (collection_attention_kinds) {
    result = f_matrix_multiply(NULL, collection_attention_kinds, weights->weight_output[d_W]);
    if (optional_internal_states) {
      if (optional_internal_states->attention_output)
        f_matrix_free(optional_internal_states->attention_output);
      optional_internal_states->attention_output = collection_attention_kinds;
    } else {
      f_matrix_free(collection_attention_kinds);
    }
  }
  if (!optional_internal_states) {
    f_matrix_free(scoped_attention_weights);
    f_matrix_free(scores);
    f_matrix_free(sequence_tokens_query);
    f_matrix_free(sequence_tokens_key);
    f_matrix_free(sequence_tokens_value);
  }
  return result;
}
/* NARDINAN chapter:
 * now we're going to integrate the classic feedforward components. The
 * attention gave us information about how the components are linked together,
 * but now we need to "think" with a classic feedforward approach.
 */
#define d_feedforward_size (d_model * 4) /* another constant to define the size of the model */
typedef struct s_feedforward_weights {
  /* the weight hidden is [d_model x d_feedforward_size] while bias is [1 x
   * d_feedforward_size]. The output weight is [d_feedforwad_size x d_model] and
   * the final bial is [1 x d_model], as you can obviously imagine.
   */
  t_matrix *weight_hidden[2], *bias_hidden[2], *weight_output[2], *bias_output[2]; /* well, a standard 1-hidden-layer FFN. What else? */
} s_feedforward_weights;
void f_feedforward_weights_free(s_feedforward_weights *weights) {
  if (weights) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      if (weights->weight_hidden[index_matrix])
        f_matrix_free(weights->weight_hidden[index_matrix]);
      if (weights->weight_output[index_matrix])
        f_matrix_free(weights->weight_output[index_matrix]);
      if (weights->bias_hidden[index_matrix])
        f_matrix_free(weights->bias_hidden[index_matrix]);
      if (weights->bias_output[index_matrix])
        f_matrix_free(weights->bias_output[index_matrix]);
    }
    free(weights);
  }
}
s_feedforward_weights *f_feedforward_weights_new(void) {
  s_feedforward_weights *result = (s_feedforward_weights *) malloc(sizeof(s_feedforward_weights));
  if (result) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      result->weight_hidden[index_matrix] = f_matrix_new(d_model, d_feedforward_size);
      result->bias_hidden[index_matrix] = f_matrix_new(1, d_feedforward_size);
      result->weight_output[index_matrix] = f_matrix_new(d_feedforward_size, d_model);
      result->bias_output[index_matrix] = f_matrix_new(1, d_model);
    }
    if ((result->weight_hidden[d_W]) && (result->weight_hidden[d_G]) && (result->bias_hidden[d_W]) && (result->bias_hidden[d_G]) &&
        (result->weight_output[d_W]) && (result->weight_output[d_G]) && (result->bias_output[d_W]) && (result->bias_output[d_G])) {
      for (size_t index_column = 0; index_column < d_feedforward_size; ++index_column) {
        for (size_t index_row = 0; index_row < d_model; ++index_row)
          d_matrix_getCR(result->weight_hidden[d_W], index_column, index_row) = f_random_float();
        d_matrix_getCR(result->bias_hidden[d_W], index_column, 0) = 0;
      }
      for (size_t index_column = 0; index_column < d_model; ++index_column) {
        for (size_t index_row = 0; index_row < d_feedforward_size; ++index_row)
          d_matrix_getCR(result->weight_output[d_W], index_column, index_row) = f_random_float();
        d_matrix_getCR(result->bias_output[d_W], index_column, 0) = 0;
      }
    } else {
      f_feedforward_weights_free(result);
      result = NULL;
    }
  }
  return result;
}
void f_feedforward_weights_gradient_zero(s_feedforward_weights *weights) {
  f_matrix_zero(weights->weight_hidden[d_G]);
  f_matrix_zero(weights->bias_hidden[d_G]);
  f_matrix_zero(weights->weight_output[d_G]);
  f_matrix_zero(weights->bias_output[d_G]);
}
float f_GELU(float value) {
  /* the gaussian-error-linear-unit is a smooth non-linear distribution that
   * goes form 0 to THE UNIVERSE! Better the ReLU for this reason: <provide
   * reason, I don't know, just testing> (well, almost copy-pasted from the
   * paper)
   */
  return 0.5f * value * (1.0f + tanhf(0.7978845608f * (value + 0.044715f * value * value * value)));
}
float sech_squared(float value) { /* hyperbolic secant function */
  float tanhf_value = tanhf(value);
  return (1 - (tanhf_value * tanhf_value));
}
float f_GELU_derivative(float value) {
  return (0.5f * (1.0f + tanhf(0.7978845608f * (value + 0.044715f * value * value * value)))) +
      (0.5f * value * sech_squared(0.7978845608f * (value + 0.044715f * value * value * value)) * 0.7978845608f * (1 + 3.0f * 0.044715 * value * value));
}
t_matrix *f_feedforward_encoded_output_new(s_transformer_internal_states *optional_internal_states, s_feedforward_weights *weights,
    t_matrix *embedded_sequence_with_attention_weights) {
  t_matrix *hidden_layer = NULL, *result = NULL;
  if (optional_internal_states) {
    if (optional_internal_states->feedforward_hidden_layer_pre_GELU)
      f_matrix_free(optional_internal_states->feedforward_hidden_layer_pre_GELU);
    hidden_layer = optional_internal_states->feedforward_hidden_layer_pre_GELU = f_matrix_multiply(NULL, embedded_sequence_with_attention_weights,
        weights->weight_hidden[d_W]);
  } else {
    hidden_layer = f_matrix_multiply(NULL, embedded_sequence_with_attention_weights, weights->weight_hidden[d_W]);
  }
  if (hidden_layer) {
    for (size_t index_column = 0; index_column < f_matrix_columns(hidden_layer); ++index_column)
      for (size_t index_row = 0; index_row < f_matrix_rows(hidden_layer); ++index_row)
        d_matrix_getCR(hidden_layer, index_column, index_row) = d_matrix_getCR(hidden_layer, index_column, index_row) +
            d_matrix_getCR(weights->bias_hidden[d_W], index_column, 0);
    if (optional_internal_states) {
      if (optional_internal_states->feedforward_hidden_layer_post_GELU)
        f_matrix_free(optional_internal_states->feedforward_hidden_layer_post_GELU);
      hidden_layer = optional_internal_states->feedforward_hidden_layer_post_GELU =
          f_matrix_duplicate(optional_internal_states->feedforward_hidden_layer_pre_GELU);
    }
    for (size_t index_column = 0; index_column < f_matrix_columns(hidden_layer); ++index_column)
      for (size_t index_row = 0; index_row < f_matrix_rows(hidden_layer); ++index_row)
        d_matrix_getCR(hidden_layer, index_column, index_row) = f_GELU(d_matrix_getCR(hidden_layer, index_column, index_row));
    if ((result = f_matrix_multiply(NULL, hidden_layer, weights->weight_output[d_W])))
      for (size_t index_column = 0; index_column < f_matrix_columns(result); ++index_column)
        for (size_t index_row = 0; index_row < f_matrix_rows(result); ++index_row)
          d_matrix_getCR(result, index_column, index_row) = d_matrix_getCR(result, index_column, index_row) +
              d_matrix_getCR(weights->bias_output[d_W], index_column, 0);
    if (!optional_internal_states)
      f_matrix_free(hidden_layer);
  }
  return result;
}
/* NARDINAN chapter:
 * now we're going to prevent gradient explosion or vanishing with our layer
 * normalisation process. We use a gamma and beta vectors to perform this
 * normalization */
typedef struct s_layer_normalization_weights {
  /* gamma and beta are the vectors that are telling us what's the gradient (for
   * how much we need to multiply) and what's the offset. These are giving us a
   * sort of "magnitude" and "direction" (like a vector) to the normalized
   * values after the layer normalization (each value is going to be mean 0 and
   * variance 1). Their size is the same: [1 x d_model]
   */
  t_matrix *gamma_weights[2], *beta_weights[2];
} s_layer_normalization_weights;
void f_layer_normalization_weights_free(s_layer_normalization_weights *weights) {
  if (weights) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      if (weights->gamma_weights[index_matrix])
        f_matrix_free(weights->gamma_weights[index_matrix]);
      if (weights->beta_weights[index_matrix])
        f_matrix_free(weights->beta_weights[index_matrix]);
    }
    free(weights);
  }
}
s_layer_normalization_weights *f_layer_normalization_weights_new(void) {
  s_layer_normalization_weights *result = (s_layer_normalization_weights *) malloc(sizeof(s_layer_normalization_weights));
  if (result) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      result->gamma_weights[index_matrix] = f_matrix_new(1, d_model);
      result->beta_weights[index_matrix] = f_matrix_new(1, d_model);
    }
    if ((result->beta_weights[d_W]) && (result->beta_weights[d_G]) && (result->gamma_weights[d_W]) && (result->gamma_weights[d_G])) {
      for (size_t index_column = 0; index_column < d_model; ++index_column) {
        d_matrix_getCR(result->gamma_weights[d_W], index_column, 0) = 1.0; /* we're neural, we star by multiplying by 1 ... */
        d_matrix_getCR(result->beta_weights[d_W], index_column, 0) = 0.0; /* ... and by applying an offset of zero */
      }
    } else {
      f_layer_normalization_weights_free(result);
      result = NULL;
    }
  }
  return result;
}
void f_layer_normalization_weights_gradient_zero(s_layer_normalization_weights *weights) {
  f_matrix_zero(weights->beta_weights[d_G]);
  f_matrix_zero(weights->gamma_weights[d_G]);
}
t_matrix *f_layer_normalization_forward_new(s_layer_normalization_weights *weights, t_matrix *input_matrix) {
  /* this is a util, it is going to normalize the input matrix */
  d_assert(f_matrix_columns(input_matrix) == d_model);
  t_matrix *result = f_matrix_new(f_matrix_rows(input_matrix), f_matrix_columns(input_matrix));
  if (result)
    for (size_t index_row = 0; index_row < f_matrix_rows(input_matrix); ++index_row) {
      float mean = 0, variance = 0, sqrt_variance = 0;
      for (size_t index_column = 0; index_column < d_model; ++index_column)
        mean += d_matrix_getCR(input_matrix, index_column, index_row);
      mean /= (float) d_model;
      for (size_t index_column = 0; index_column < d_model; ++index_column)
        variance += ((d_matrix_getCR(input_matrix, index_column, index_row) - mean) * (d_matrix_getCR(input_matrix, index_column, index_row) - mean));
      variance /= (float) d_model;
      sqrt_variance = sqrtf(variance + 1e-5 /* just to avoid getting zero as result */);
      /* we're going to normalize each row independently: first we get the
       * normalized value and then, each value, it is multiplied by the gamma
       * weights. We're going to add - as offset out beta weights */
      for (size_t index_column = 0; index_column < d_model; ++index_column)
        d_matrix_getCR(result, index_column, index_row) = (((d_matrix_getCR(input_matrix, index_column, index_row) - mean) / sqrt_variance) *
                                                              d_matrix_getCR(weights->gamma_weights[d_W], index_column, 0)) +
            d_matrix_getCR(weights->beta_weights[d_W], index_column, 0);
    }
  return result;
}
/* NARDINAN chapter:
 * now we need to put everything together: what we've seen so far is A SINGLE
 * layer of our transformer. We might need to go through d_number_layers of them
 */
#define d_number_layers 4
typedef struct s_transformer_weights {
  s_layer_normalization_weights *before_attention_weights, *before_feedforward_weights;
  s_attention_weights *attention_weights;
  s_feedforward_weights *feedforward_weights;
} s_transformer_weights;
void f_transformer_weights_free(s_transformer_weights *weights) {
  if (weights) {
    /* probably here we have to dump them on the disk, to prevent data-loss */
    f_layer_normalization_weights_free(weights->before_attention_weights);
    f_layer_normalization_weights_free(weights->before_feedforward_weights);
    f_attention_weights_free(weights->attention_weights);
    f_feedforward_weights_free(weights->feedforward_weights);
    free(weights);
  }
}
s_transformer_weights *f_transformer_weights_new(void) {
  s_transformer_weights *result = (s_transformer_weights *) malloc(sizeof(s_transformer_weights));
  if (result) {
    /* load the data if the model exists, otherwise start from scratch */
    result->before_attention_weights = f_layer_normalization_weights_new();
    result->before_feedforward_weights = f_layer_normalization_weights_new();
    result->attention_weights = f_attention_weights_new();
    result->feedforward_weights = f_feedforward_weights_new();
    if ((!result->before_attention_weights) || (!result->before_feedforward_weights) || (!result->attention_weights) || (!result->feedforward_weights)) {
      f_transformer_weights_free(result);
      result = NULL;
    }
  }
  return result;
}
void f_transformer_weights_gradient_zero(s_transformer_weights *weights) {
  f_layer_normalization_weights_gradient_zero(weights->before_attention_weights);
  f_layer_normalization_weights_gradient_zero(weights->before_feedforward_weights);
  f_attention_weights_gradient_zero(weights->attention_weights);
  f_feedforward_weights_gradient_zero(weights->feedforward_weights);
}
t_matrix *f_encoded_transformer_weights_forward(s_transformer_internal_states *optional_internal_states, s_transformer_weights *weights,
    t_matrix *embedded_sequence) {
  /* The classic process, similar to a feedforward but we're going to encode
   * with the attention the normalized verision, and we're going to do the same
   * thing for the feedforward fuction. This means that we're going to get:
   *
   * attention = embedded_sequence + attention(normalized(embedded_sequence)) -
   * this is a way to prevent exploding/vanishing gradient result = attention +
   * feedforward(normalized(attention)) - same, as above.
   */
  t_matrix *embedded_sequence_normalized, *result = NULL;
  if ((embedded_sequence_normalized = f_layer_normalization_forward_new(weights->before_attention_weights, embedded_sequence))) {
    t_matrix *final_attention_weights_table;
    if ((final_attention_weights_table = f_encoded_attention_weights_forward_new(optional_internal_states, weights->attention_weights,
             embedded_sequence_normalized))) {
      /* we got the attention, now it is time for the feedforward */
      t_matrix *result_normalized;
      f_matrix_add_in_place(final_attention_weights_table, embedded_sequence);
      if ((result_normalized = f_layer_normalization_forward_new(weights->before_feedforward_weights, final_attention_weights_table))) {
        if ((result = f_feedforward_encoded_output_new(optional_internal_states, weights->feedforward_weights, result_normalized)))
          f_matrix_add_in_place(result, final_attention_weights_table);
        if (optional_internal_states) {
          if (optional_internal_states->before_feedforward_normalized)
            f_matrix_free(optional_internal_states->before_feedforward_normalized);
          optional_internal_states->before_feedforward_normalized = result_normalized;
        } else {
          f_matrix_free(result_normalized);
        }
      }
      if (optional_internal_states) {
        if (optional_internal_states->after_attention_normalized)
          f_matrix_free(optional_internal_states->after_attention_normalized);
        optional_internal_states->after_attention_normalized = final_attention_weights_table;
      } else {
        f_matrix_free(final_attention_weights_table);
      }
    }
    if (optional_internal_states) {
      if (optional_internal_states->before_attention_normalized)
        f_matrix_free(optional_internal_states->before_attention_normalized);
      optional_internal_states->before_attention_normalized = embedded_sequence_normalized;
    } else {
      f_matrix_free(embedded_sequence_normalized);
    }
  }
  return result;
}
typedef struct s_GPT_model {
  t_matrix *embedding_table[2], *positional_encoding_table, *new_iteration_head[2], *final_normalized_sequence, *embedded_sequence;
  s_transformer_weights *transformer_weights[d_number_layers];
  s_layer_normalization_weights *final_normalization_weights;
  s_transformer_internal_states internal_states[d_number_layers];
} s_GPT_model;
void f_GPT_model_free(s_GPT_model *model) {
  if (model) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      f_matrix_free(model->embedding_table[index_matrix]);
      f_matrix_free(model->new_iteration_head[index_matrix]);
    }
    f_matrix_free(model->positional_encoding_table);
    f_matrix_free(model->final_normalized_sequence);
    f_matrix_free(model->embedded_sequence);
    for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer) {
      f_transformer_weights_free(model->transformer_weights[index_layer]);
      f_transformer_internal_states_free(&(model->internal_states[index_layer]));
    }
    f_layer_normalization_weights_free(model->final_normalization_weights);
    free(model);
  }
}
s_GPT_model *f_GPT_model_new(size_t context_length) {
  s_GPT_model *result = (s_GPT_model *) malloc(sizeof(s_GPT_model));
  if (result) {
    /* we need to load the file */
    memset(result->internal_states, 0, (sizeof(s_transformer_internal_states) * d_number_layers));
    result->final_normalized_sequence = NULL;
    result->embedded_sequence = NULL;
    result->embedding_table[d_W] = f_embedding_table_new();
    result->embedding_table[d_G] = f_matrix_new(d_vocabulary_size, d_model);
    result->positional_encoding_table = f_positional_encoding_table_new(context_length);
    if ((result->new_iteration_head[d_W] = f_matrix_new(d_model, d_vocabulary_size))) {
      result->new_iteration_head[d_G] = f_matrix_new(d_model, d_vocabulary_size);
      for (size_t index_row = 0; index_row < d_model; ++index_row)
        for (size_t index_column = 0; index_column < d_vocabulary_size; ++index_column)
          d_matrix_getCR(result->new_iteration_head[d_W], index_column, index_row) = f_random_float();
      for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer)
        result->transformer_weights[index_layer] = f_transformer_weights_new();
      result->final_normalization_weights = f_layer_normalization_weights_new();
    }
    if ((result->embedding_table[d_W]) && (result->embedding_table[d_G]) && (result->new_iteration_head[d_W]) && (result->new_iteration_head[d_G]) &&
        (result->positional_encoding_table) && (result->final_normalization_weights)) {
      for (size_t index_layer = 0; (index_layer < d_number_layers) && (result); ++index_layer)
        if (!result->transformer_weights[index_layer]) {
          f_GPT_model_free(result);
          result = NULL;
        }
    } else {
      f_GPT_model_free(result);
      result = NULL;
    }
  }
  return result;
}
void f_GPT_model_gradient_zero(s_GPT_model *model) {
  f_matrix_zero(model->embedding_table[d_G]);
  f_matrix_zero(model->new_iteration_head[d_G]);
  f_matrix_zero(model->final_normalization_weights->gamma_weights[d_G]);
  f_matrix_zero(model->final_normalization_weights->beta_weights[d_G]);
  for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer)
    f_transformer_weights_gradient_zero(model->transformer_weights[index_layer]);
}
/* now the real forward! */
t_matrix *f_GPT_model_forward_new(s_GPT_model *model, const int *tokens, size_t length_encoded) {
  t_matrix *embedded_sequence = f_embedded_sequence_new(model->embedding_table[d_W], model->positional_encoding_table, tokens, length_encoded), *result = NULL;
  if (embedded_sequence) {
    f_matrix_free(model->internal_states[0].output);
    if ((model->internal_states[0].output = f_encoded_transformer_weights_forward(&(model->internal_states[0]), model->transformer_weights[0],
             embedded_sequence))) {
      for (size_t index_layer = 1; (index_layer < d_number_layers) && (model->internal_states[index_layer - 1].output); ++index_layer) {
        f_matrix_free(model->internal_states[index_layer].output);
        model->internal_states[index_layer].output = f_encoded_transformer_weights_forward(&(model->internal_states[index_layer]),
            model->transformer_weights[index_layer], model->internal_states[index_layer - 1].output);
      }
      f_matrix_free(model->final_normalized_sequence);
      if ((model->internal_states[d_number_layers - 1].output) &&
          ((model->final_normalized_sequence = f_layer_normalization_forward_new(model->final_normalization_weights,
                model->internal_states[d_number_layers - 1].output))))
        result = f_matrix_multiply(NULL, model->final_normalized_sequence, model->new_iteration_head[d_W]);
    }
    f_matrix_free(model->embedded_sequence);
    model->embedded_sequence = embedded_sequence;
  }
  return result;
}
/* NARDINAN chapter
 * It is time now to compute the error of this final forward matrix. I'm going
 * to use basically the same concepts I've also used in the standard RNN and FNN
 */
/* if masks is null, the logic is applied on everything, otherwise only on targets where masks[index_sqeuence] == 1 */
float f_cross_entropy_loss(t_matrix *model_forwarded_matrix, const int *targets, const bool *masks, size_t length) {
  d_assert((f_matrix_rows(model_forwarded_matrix) >= length) && (f_matrix_columns(model_forwarded_matrix) == d_vocabulary_size));
  float sum_losses_in_time = 0;
  size_t active_targets = 0;
  for (size_t index_sequence = 0; index_sequence < length; ++index_sequence)
    if ((!masks) || (masks[index_sequence])) {
      f_vector_log_softmax((t_vector *) &(d_matrix_getCR(model_forwarded_matrix, 0, index_sequence)), d_vocabulary_size);
      sum_losses_in_time += d_matrix_getCR(model_forwarded_matrix, targets[index_sequence], index_sequence);
      ++active_targets;
    }
  if (!active_targets)
    active_targets = 1;
  return -(1 / (float) active_targets) * sum_losses_in_time;
}
/* Let's compute the gradients of the loss directly in place */
void f_cross_entropy_loss_backward(t_matrix *model_forwarded_matrix, const int *target, const bool *masks, size_t length) {
  d_assert((f_matrix_rows(model_forwarded_matrix) >= length) && (f_matrix_columns(model_forwarded_matrix) == d_vocabulary_size));
  size_t active_targets = length;
  if (masks) {
    active_targets = 0;
    for (size_t index_sequence = 0; index_sequence < length; ++index_sequence)
      active_targets += masks[index_sequence];
  }
  if (active_targets > 0)
    for (size_t index_sequence = 0; index_sequence < length; ++index_sequence) {
      if ((!masks) || (masks[index_sequence])) {
        for (size_t index_column = 0; index_column < d_vocabulary_size; ++index_column)
          d_matrix_getCR(model_forwarded_matrix, index_column, index_sequence) = (expf(d_matrix_getCR(model_forwarded_matrix, index_column, index_sequence)) -
                                                                                     ((target[index_sequence] == index_column) ? 1.0 : 0.0)) /
              (float) active_targets;
      } else {
        /* of course, if we're not masked we should set everything to zero, JC! */
        for (size_t index_column = 0; index_column < d_vocabulary_size; ++index_column)
          d_matrix_getCR(model_forwarded_matrix, index_column, index_sequence) = 0;
      }
    }
}
/* we now compute the gradient WRT the weights, and then the gradient WRT the
 * input and we return it */
t_matrix *f_GPT_model_head_backward_new(s_GPT_model *model, t_matrix *output_gradients, t_matrix *normalized_sequence) {
  t_matrix *gradient_WRT_weights = f_matrix_multiply_transposed_with_matrix(NULL, normalized_sequence, output_gradients), *result = NULL;
  if (gradient_WRT_weights) {
    f_matrix_add_in_place(model->new_iteration_head[d_G], gradient_WRT_weights);
    result = f_matrix_multiply_matrix_with_transposed(NULL, output_gradients, model->new_iteration_head[d_W]);
    f_matrix_free(gradient_WRT_weights);
  }
  return result;
}
t_matrix *f_layer_normalization_backward_new(s_layer_normalization_weights *weights, t_matrix *output_gradients, t_matrix *input_matrix) {
  t_matrix *result = NULL;
  /* First, we compute the gradient w.r.t. gamma and gradient w.r.t. beta and we
   * store them into the weights (we use the d_G matrices to store it, quite
   * easy and it is going to be a [1, d_model] matrix. To do this, we need to be
   * sure that we can obtain the normalized input. Once this is done (just a
   * simple two-loop) we compute the gradient W.R.T. the input (and this is a
   * little bit more complicated)
   */
  if ((result = f_matrix_new(f_matrix_rows(output_gradients), d_model)))
    for (size_t index_row = 0; index_row < f_matrix_rows(output_gradients); ++index_row) {
      float mean = 0, variance = 0, sqrt_variance = 0, normalized_gradient[d_model] = {0}, normalized_value[d_model] = {0}, mean_normalized_gradient = 0,
            mean_normalized_gradient_by_value = 0;
      for (size_t index_column = 0; index_column < d_model; ++index_column)
        mean += d_matrix_getCR(input_matrix, index_column, index_row);
      mean /= (float) d_model;
      for (size_t index_column = 0; index_column < d_model; ++index_column)
        variance += ((d_matrix_getCR(input_matrix, index_column, index_row) - mean) * (d_matrix_getCR(input_matrix, index_column, index_row) - mean));
      variance /= (float) d_model;
      sqrt_variance = sqrtf(variance + 1e-5 /* just to avoid getting zero as result */);
      for (size_t index_column = 0; index_column < d_model; ++index_column) {
        d_matrix_getCR(weights->gamma_weights[d_G], index_column, 0) += d_matrix_getCR(output_gradients, index_column, index_row) *
            ((d_matrix_getCR(input_matrix, index_column, index_row) - mean) / sqrt_variance);
        d_matrix_getCR(weights->beta_weights[d_G], index_column, 0) += d_matrix_getCR(output_gradients, index_column, index_row);
      }
      for (size_t index_column = 0; index_column < d_model; ++index_column) {
        normalized_gradient[index_column] = d_matrix_getCR(output_gradients, index_column, index_row) *
            d_matrix_getCR(weights->gamma_weights[d_W], index_column, 0);
        normalized_value[index_column] = (d_matrix_getCR(input_matrix, index_column, index_row) - mean) / sqrt_variance;
        mean_normalized_gradient += normalized_gradient[index_column];
        mean_normalized_gradient_by_value += normalized_gradient[index_column] * normalized_value[index_column];
      }
      mean_normalized_gradient /= (float) d_model;
      mean_normalized_gradient_by_value /= (float) d_model;
      for (size_t index_column = 0; index_column < d_model; ++index_column)
        d_matrix_getCR(result, index_column, index_row) = (1 / sqrt_variance) *
            (normalized_gradient[index_column] - mean_normalized_gradient - (normalized_value[index_column] * mean_normalized_gradient_by_value));
    }
  return result;
}
t_matrix *f_feedforward_backward_new(s_transformer_internal_states *states, s_feedforward_weights *weights, t_matrix *output_gradients) {
  t_matrix *gradients_WRT_output_weights = f_matrix_multiply_transposed_with_matrix(NULL, states->feedforward_hidden_layer_post_GELU, output_gradients),
           *gradients_WRT_post_GELU = f_matrix_multiply_matrix_with_transposed(NULL, output_gradients, weights->weight_output[d_W]), *result = NULL;
  if ((gradients_WRT_output_weights) && (gradients_WRT_post_GELU)) {
    t_matrix *gradients_WRT_pre_GELU = f_matrix_new(f_matrix_rows(states->feedforward_hidden_layer_pre_GELU),
        f_matrix_columns(states->feedforward_hidden_layer_pre_GELU));
    /* we need to migrate gradients_WRT_ouput_weights into weight_output */
    f_matrix_add_in_place(weights->weight_output[d_G], gradients_WRT_output_weights);
    for (size_t index_column = 0; index_column < f_matrix_columns(output_gradients); ++index_column)
      for (size_t index_row = 0; index_row < f_matrix_rows(output_gradients); ++index_row)
        d_matrix_getCR(weights->bias_output[d_G], index_column, 0) += d_matrix_getCR(output_gradients, index_column, index_row);
    if (gradients_WRT_pre_GELU) {
      t_matrix *gradients_WRT_hidden_weights;
      /* now it's time for the hidden gradients for both bias and value */
      for (size_t index_row = 0; index_row < f_matrix_rows(gradients_WRT_pre_GELU); ++index_row)
        for (size_t index_column = 0; index_column < f_matrix_columns(gradients_WRT_pre_GELU); ++index_column) {
          d_matrix_getCR(gradients_WRT_pre_GELU, index_column, index_row) = d_matrix_getCR(gradients_WRT_post_GELU, index_column, index_row) *
              f_GELU_derivative(d_matrix_getCR(states->feedforward_hidden_layer_pre_GELU, index_column, index_row));
        }
      for (size_t index_column = 0; index_column < f_matrix_columns(gradients_WRT_post_GELU); ++index_column)
        for (size_t index_row = 0; index_row < f_matrix_rows(gradients_WRT_pre_GELU); ++index_row)
          d_matrix_getCR(weights->bias_hidden[d_G], index_column, 0) += d_matrix_getCR(gradients_WRT_pre_GELU, index_column, index_row);
      if ((gradients_WRT_hidden_weights = f_matrix_multiply_transposed_with_matrix(NULL, states->before_feedforward_normalized, gradients_WRT_pre_GELU))) {
        f_matrix_add_in_place(weights->weight_hidden[d_G], gradients_WRT_hidden_weights);
        result = f_matrix_multiply_matrix_with_transposed(NULL, gradients_WRT_pre_GELU, weights->weight_hidden[d_W]);
        f_matrix_free(gradients_WRT_hidden_weights);
      }
      f_matrix_free(gradients_WRT_pre_GELU);
    }
  }
  f_matrix_free(gradients_WRT_output_weights);
  f_matrix_free(gradients_WRT_post_GELU);
  return result;
}
t_matrix *f_attention_backward_new(s_transformer_internal_states *states, s_attention_weights *weights, t_matrix *output_gradients) {
  t_matrix *gradients_WRT_output_weights = f_matrix_multiply_transposed_with_matrix(NULL, states->attention_output, output_gradients), *result = NULL;
  if (gradients_WRT_output_weights) {
    t_matrix *gradients_WRT_attention_concatenated = f_matrix_multiply_matrix_with_transposed(NULL, output_gradients, weights->weight_output[d_W]);
    if (gradients_WRT_attention_concatenated) {
      t_matrix *submatrix_gradient_WRT_attention_concatenated = f_matrix_new(f_matrix_rows(output_gradients), d_size_head);
      f_matrix_add_in_place(weights->weight_output[d_G], gradients_WRT_output_weights);
      if (submatrix_gradient_WRT_attention_concatenated) {
        t_matrix *gradients_WRT_value = NULL, *gradients_WRT_scores_after_softmax = NULL, *gradients_WRT_scores_before_softmax = NULL,
                 *gradients_WRT_query = NULL, *gradients_WRT_key = NULL;
        const float sqrt_size_head = sqrtf((float) d_size_head);
        /* we'll put in the submatrix_gradient_WRT_attention_concatenated only
         * the submatrix with the d_size_head columns of the head pointed out */
        for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
          for (size_t index_size_head = (index_head * d_size_head); index_size_head < ((index_head + 1) * d_size_head); ++index_size_head)
            for (size_t index_row = 0; index_row < f_matrix_rows(output_gradients); ++index_row)
              d_matrix_getCR(submatrix_gradient_WRT_attention_concatenated, (index_size_head - (index_head * d_size_head)),
                  index_row) = d_matrix_getCR(gradients_WRT_attention_concatenated, index_size_head, index_row);
          if ((gradients_WRT_value = f_matrix_multiply_transposed_with_matrix(gradients_WRT_value, states->scores[index_head],
                   submatrix_gradient_WRT_attention_concatenated))) {
            if ((gradients_WRT_scores_after_softmax = f_matrix_multiply_matrix_with_transposed(gradients_WRT_scores_after_softmax,
                     submatrix_gradient_WRT_attention_concatenated, states->value[index_head]))) {
              if ((gradients_WRT_scores_before_softmax) ||
                  ((gradients_WRT_scores_before_softmax = f_matrix_new(f_matrix_rows(output_gradients), f_matrix_rows(output_gradients))))) {
                for (size_t index_row = 0; index_row < f_matrix_rows(output_gradients); ++index_row) {
                  float dot_product = 0;
                  for (size_t index_column = 0; index_column < f_matrix_rows(output_gradients); ++index_column)
                    dot_product += (d_matrix_getCR(states->scores[index_head], index_column, index_row) *
                        d_matrix_getCR(gradients_WRT_scores_after_softmax, index_column, index_row));
                  for (size_t index_column = 0; index_column < f_matrix_rows(output_gradients); ++index_column) {
                    d_matrix_getCR(gradients_WRT_scores_before_softmax, index_column,
                        index_row) = (d_matrix_getCR(states->scores[index_head], index_column, index_row) *
                                         (d_matrix_getCR(gradients_WRT_scores_after_softmax, index_column, index_row) - dot_product)) /
                        sqrt_size_head;
                  }
                }
                /* and now, a long sequence of multiplications to update the
                 * final result, that is the gradient WRT before attention
                 * normalized (well, the input my friend)  */
                if (((gradients_WRT_query = f_matrix_multiply(gradients_WRT_query, gradients_WRT_scores_before_softmax, states->key[index_head]))) &&
                    ((gradients_WRT_key = f_matrix_multiply_transposed_with_matrix(gradients_WRT_key, gradients_WRT_scores_before_softmax,
                          states->query[index_head])))) {
                  f_matrix_multiply_transposed_with_matrix_add_into_host(weights->weight_query[index_head][d_G], states->before_attention_normalized,
                      gradients_WRT_query);
                  f_matrix_multiply_transposed_with_matrix_add_into_host(weights->weight_key[index_head][d_G], states->before_attention_normalized,
                      gradients_WRT_key);
                  f_matrix_multiply_transposed_with_matrix_add_into_host(weights->weight_value[index_head][d_G], states->before_attention_normalized,
                      gradients_WRT_value);
                  if ((result) || ((result = f_matrix_new(f_matrix_rows(output_gradients), f_matrix_columns(output_gradients))))) {
                    f_matrix_multiply_matrix_with_transposed_add_into_host(result, gradients_WRT_query, weights->weight_query[index_head][d_W]);
                    f_matrix_multiply_matrix_with_transposed_add_into_host(result, gradients_WRT_key, weights->weight_key[index_head][d_W]);
                    f_matrix_multiply_matrix_with_transposed_add_into_host(result, gradients_WRT_value, weights->weight_value[index_head][d_W]);
                  }
                }
              }
            }
          }
        }
        f_matrix_free(gradients_WRT_key);
        f_matrix_free(gradients_WRT_query);
        f_matrix_free(gradients_WRT_scores_before_softmax);
        f_matrix_free(gradients_WRT_scores_after_softmax);
        f_matrix_free(gradients_WRT_value);
        f_matrix_free(submatrix_gradient_WRT_attention_concatenated);
      }
      f_matrix_free(gradients_WRT_attention_concatenated);
    }
    f_matrix_free(gradients_WRT_output_weights);
  }
  return result;
}
t_matrix *f_transformer_block_backward_new(s_transformer_internal_states *states, s_transformer_weights *weights, t_matrix *embedded_sequence,
    t_matrix *output_gradients) {
  t_matrix *feedforward_backward = f_feedforward_backward_new(states, weights->feedforward_weights, output_gradients), *result = NULL;
  if (feedforward_backward) {
    t_matrix *normalized_feedforward_backward = f_layer_normalization_backward_new(weights->before_feedforward_weights, feedforward_backward,
        states->after_attention_normalized);
    if (normalized_feedforward_backward) {
      t_matrix *attention_backward;
      f_matrix_add_in_place(normalized_feedforward_backward, output_gradients);
      if ((attention_backward = f_attention_backward_new(states, weights->attention_weights, normalized_feedforward_backward))) {
        if ((result = f_layer_normalization_backward_new(weights->before_attention_weights, attention_backward, embedded_sequence))) {
          f_matrix_add_in_place(result, normalized_feedforward_backward);
        }
        f_matrix_free(attention_backward);
      }
      f_matrix_free(normalized_feedforward_backward);
    }
    f_matrix_free(feedforward_backward);
  }
  return result;
}
/* There's nothing we can do: to be sure that the gradients keep some memory on what they're doing, we need to keep track for each weight matrix in the model
 * of its M and V (basically, M keeps track of the direction of hte gradients, V of their magnitude) */
#define d_M 0 /* where are gradients been pointing, lately ? */
#define d_V 1 /* how's the magnitude of those gradients been, lately ?*/
typedef struct s_layer_normalization_optimizer_state {
  t_matrix *gamma_weights[2], *beta_weights[2];
} s_layer_normalization_optimizer_state;
void f_layer_normalization_optimizer_state_free(s_layer_normalization_optimizer_state *state) {
  if (state) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      if (state->gamma_weights[index_matrix])
        f_matrix_free(state->gamma_weights[index_matrix]);
      if (state->beta_weights[index_matrix])
        f_matrix_free(state->beta_weights[index_matrix]);
    }
    free(state);
  }
}
s_layer_normalization_optimizer_state *f_layer_normalization_optimizer_state_new(void) {
  s_layer_normalization_optimizer_state *result = (s_layer_normalization_optimizer_state *) malloc(sizeof(s_layer_normalization_optimizer_state));
  if (result) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      result->gamma_weights[index_matrix] = f_matrix_new(1, d_model);
      result->beta_weights[index_matrix] = f_matrix_new(1, d_model);
    }
    if ((result->gamma_weights[d_M]) && (result->gamma_weights[d_V]) && (result->beta_weights[d_M]) && (result->beta_weights[d_V])) {
      for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
        f_matrix_zero(result->gamma_weights[index_matrix]);
        f_matrix_zero(result->beta_weights[index_matrix]);
      }
    } else {
      f_layer_normalization_optimizer_state_free(result);
      result = NULL;
    }
  }
  return result;
}
typedef struct s_attention_optimizer_state {
  t_matrix *weight_query[d_number_heads][2], *weight_key[d_number_heads][2], *weight_value[d_number_heads][2], *weight_output[2];
} s_attention_optimizer_state;
void f_attention_optimizer_state_free(s_attention_optimizer_state *state) {
  if (state) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
        f_matrix_free(state->weight_query[index_head][index_matrix]);
        f_matrix_free(state->weight_key[index_head][index_matrix]);
        f_matrix_free(state->weight_value[index_head][index_matrix]);
      }
      f_matrix_free(state->weight_output[index_matrix]);
    }
    free(state);
  }
}
s_attention_optimizer_state *f_attention_optimizer_state_new(void) {
  s_attention_optimizer_state *result = (s_attention_optimizer_state *) malloc(sizeof(s_attention_optimizer_state));
  for (size_t index_head = 0; (result) && (index_head < d_number_heads); ++index_head) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      result->weight_query[index_head][index_matrix] = f_matrix_new(d_model, d_size_head);
      result->weight_key[index_head][index_matrix] = f_matrix_new(d_model, d_size_head);
      result->weight_value[index_head][index_matrix] = f_matrix_new(d_model, d_size_head);
    }
    if ((result->weight_query[index_head][d_M]) && (result->weight_query[index_head][d_V]) && (result->weight_key[index_head][d_M]) &&
        (result->weight_key[index_head][d_V]) && (result->weight_value[index_head][d_M]) && (result->weight_value[index_head][d_V])) {
      for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
        f_matrix_zero(result->weight_query[index_head][index_matrix]);
        f_matrix_zero(result->weight_key[index_head][index_matrix]);
        f_matrix_zero(result->weight_value[index_head][index_matrix]);
      }
    } else {
      f_attention_optimizer_state_free(result);
      result = NULL;
    }
  }
  if (result) {
    if ((result->weight_output[d_M] = f_matrix_new(d_model, d_model)) && (result->weight_output[d_V] = f_matrix_new(d_model, d_model))) {
      f_matrix_zero(result->weight_output[d_M]);
      f_matrix_zero(result->weight_output[d_V]);
    } else {
      f_attention_optimizer_state_free(result);
      result = NULL;
    }
  }
  return result;
}
typedef struct s_feedforward_optimizer_state {
  t_matrix *weight_hidden[2], *bias_hidden[2], *weight_output[2], *bias_output[2];
} s_feedforward_optimizer_state;
void f_feedforward_optimizer_state_free(s_feedforward_optimizer_state *state) {
  if (state) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      if (state->weight_hidden[index_matrix])
        f_matrix_free(state->weight_hidden[index_matrix]);
      if (state->weight_output[index_matrix])
        f_matrix_free(state->weight_output[index_matrix]);
      if (state->bias_hidden[index_matrix])
        f_matrix_free(state->bias_hidden[index_matrix]);
      if (state->bias_output[index_matrix])
        f_matrix_free(state->bias_output[index_matrix]);
    }
    free(state);
  }
}
s_feedforward_optimizer_state *f_feedforward_optimizer_state_new(void) {
  s_feedforward_optimizer_state *result = (s_feedforward_optimizer_state *) malloc(sizeof(s_feedforward_optimizer_state));
  if (result) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      result->weight_hidden[index_matrix] = f_matrix_new(d_model, d_feedforward_size);
      result->bias_hidden[index_matrix] = f_matrix_new(1, d_feedforward_size);
      result->weight_output[index_matrix] = f_matrix_new(d_feedforward_size, d_model);
      result->bias_output[index_matrix] = f_matrix_new(1, d_model);
    }
    if ((result->weight_hidden[d_M]) && (result->weight_hidden[d_V]) && (result->bias_hidden[d_M]) && (result->bias_hidden[d_V]) &&
        (result->weight_output[d_M]) && (result->weight_output[d_V]) && (result->bias_output[d_M]) && (result->bias_output[d_V])) {
      for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
        f_matrix_zero(result->weight_hidden[index_matrix]);
        f_matrix_zero(result->bias_hidden[index_matrix]);
        f_matrix_zero(result->weight_output[index_matrix]);
        f_matrix_zero(result->bias_output[index_matrix]);
      }
    } else {
      f_feedforward_optimizer_state_free(result);
      result = NULL;
    }
  }
  return result;
}
typedef struct s_transformer_optimizer_state {
  s_layer_normalization_optimizer_state *before_attention_state, *before_feedforward_state;
  s_attention_optimizer_state *attention_state;
  s_feedforward_optimizer_state *feedforward_state;
} s_transformer_optimizer_state;
void f_transformer_optimizer_state_free(s_transformer_optimizer_state *state) {
  if (state) {
    f_layer_normalization_optimizer_state_free(state->before_attention_state);
    f_layer_normalization_optimizer_state_free(state->before_feedforward_state);
    f_attention_optimizer_state_free(state->attention_state);
    f_feedforward_optimizer_state_free(state->feedforward_state);
    free(state);
  }
}
s_transformer_optimizer_state *f_transformer_optimizer_state_new(void) {
  s_transformer_optimizer_state *result = (s_transformer_optimizer_state *) malloc(sizeof(s_transformer_optimizer_state));
  if (result) {
    result->before_attention_state = f_layer_normalization_optimizer_state_new();
    result->before_feedforward_state = f_layer_normalization_optimizer_state_new();
    result->attention_state = f_attention_optimizer_state_new();
    result->feedforward_state = f_feedforward_optimizer_state_new();
    if ((!result->before_attention_state) || (!result->before_feedforward_state) || (!result->attention_state) || (!result->feedforward_state)) {
      f_transformer_optimizer_state_free(result);
      result = NULL;
    }
  }
  return result;
}
typedef struct s_GPT_optimizer_state {
  t_matrix *embedding_table[2], *new_iteration_head[2];
  s_transformer_optimizer_state *transformer_state[d_number_layers];
  s_layer_normalization_optimizer_state *final_normalization_state;
} s_GPT_optimizer_state;
void f_GPT_optimizer_state_free(s_GPT_optimizer_state *state) {
  if (state) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      f_matrix_free(state->embedding_table[index_matrix]);
      f_matrix_free(state->new_iteration_head[index_matrix]);
    }
    for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer)
      f_transformer_optimizer_state_free(state->transformer_state[index_layer]);
    f_layer_normalization_optimizer_state_free(state->final_normalization_state);
    free(state);
  }
}
s_GPT_optimizer_state *f_GPT_optimizer_state_new(void) {
  s_GPT_optimizer_state *result = (s_GPT_optimizer_state *) malloc(sizeof(s_GPT_optimizer_state));
  if (result) {
    for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
      result->embedding_table[index_matrix] = f_matrix_new(d_vocabulary_size, d_model);
      result->new_iteration_head[index_matrix] = f_matrix_new(d_model, d_vocabulary_size);
    }
    if ((result->embedding_table[d_M]) && (result->embedding_table[d_V]) && (result->new_iteration_head[d_M]) && (result->new_iteration_head[d_V])) {
      for (size_t index_matrix = 0; index_matrix < 2; ++index_matrix) {
        f_matrix_zero(result->embedding_table[index_matrix]);
        f_matrix_zero(result->new_iteration_head[index_matrix]);
      }
      for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer)
        result->transformer_state[index_layer] = f_transformer_optimizer_state_new();
      result->final_normalization_state = f_layer_normalization_optimizer_state_new();
    }
    if ((result->embedding_table[d_M]) && (result->embedding_table[d_V]) && (result->new_iteration_head[d_M]) && (result->new_iteration_head[d_V]) &&
        (result->final_normalization_state)) {
      for (size_t index_layer = 0; (index_layer < d_number_layers) && (result); ++index_layer)
        if (!result->transformer_state[index_layer]) {
          f_GPT_optimizer_state_free(result);
          result = NULL;
        }
    } else {
      f_GPT_optimizer_state_free(result);
      result = NULL;
    }
  }
  return result;
}
float f_matrix_sum_squared_values(t_matrix *matrix) {
  float result = 0;
  if (matrix)
    for (size_t index_row = 0; index_row < f_matrix_rows(matrix); ++index_row)
      for (size_t index_column = 0; index_column < f_matrix_columns(matrix); ++index_column)
        result += (d_matrix_getCR(matrix, index_column, index_row) * d_matrix_getCR(matrix, index_column, index_row));
  return result;
}
void f_matrix_scale_values(t_matrix *matrix, float scale) {
  if (matrix)
    for (size_t index_row = 0; index_row < f_matrix_rows(matrix); ++index_row)
      for (size_t index_column = 0; index_column < f_matrix_columns(matrix); ++index_column)
        d_matrix_getCR(matrix, index_column, index_row) *= scale;
}
float f_gradients_clip(s_GPT_model *model, float maximum) {
  /* OK, again, we need to go through all the gradients (Jesus, this is going to never ends) and compute the sum of each value, squared, and check if its
   * square root is above our maximum. If that's the case, we scale down all the values
   */
  float global_gradient_normal = 0;
  global_gradient_normal += f_matrix_sum_squared_values(model->embedding_table[d_G]);
  global_gradient_normal += f_matrix_sum_squared_values(model->new_iteration_head[d_G]);
  global_gradient_normal += f_matrix_sum_squared_values(model->final_normalization_weights->beta_weights[d_G]);
  global_gradient_normal += f_matrix_sum_squared_values(model->final_normalization_weights->gamma_weights[d_G]);
  for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer) {
    global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->before_attention_weights->gamma_weights[d_G]);
    global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->before_attention_weights->beta_weights[d_G]);
    global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->before_feedforward_weights->gamma_weights[d_G]);
    global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->before_feedforward_weights->beta_weights[d_G]);
    for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
      global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->attention_weights->weight_query[index_head][d_G]);
      global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->attention_weights->weight_key[index_head][d_G]);
      global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->attention_weights->weight_value[index_head][d_G]);
    }
    global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->attention_weights->weight_output[d_G]);
    global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->feedforward_weights->weight_hidden[d_G]);
    global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->feedforward_weights->bias_hidden[d_G]);
    global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->feedforward_weights->weight_output[d_G]);
    global_gradient_normal += f_matrix_sum_squared_values(model->transformer_weights[index_layer]->feedforward_weights->bias_output[d_G]);
  }
  global_gradient_normal = sqrtf(global_gradient_normal);
  if (global_gradient_normal > maximum) {
    float scale = maximum / global_gradient_normal;
    f_matrix_scale_values(model->embedding_table[d_G], scale);
    f_matrix_scale_values(model->new_iteration_head[d_G], scale);
    f_matrix_scale_values(model->final_normalization_weights->beta_weights[d_G], scale);
    f_matrix_scale_values(model->final_normalization_weights->gamma_weights[d_G], scale);
    for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer) {
      f_matrix_scale_values(model->transformer_weights[index_layer]->before_attention_weights->gamma_weights[d_G], scale);
      f_matrix_scale_values(model->transformer_weights[index_layer]->before_attention_weights->beta_weights[d_G], scale);
      f_matrix_scale_values(model->transformer_weights[index_layer]->before_feedforward_weights->gamma_weights[d_G], scale);
      f_matrix_scale_values(model->transformer_weights[index_layer]->before_feedforward_weights->beta_weights[d_G], scale);
      for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
        f_matrix_scale_values(model->transformer_weights[index_layer]->attention_weights->weight_query[index_head][d_G], scale);
        f_matrix_scale_values(model->transformer_weights[index_layer]->attention_weights->weight_key[index_head][d_G], scale);
        f_matrix_scale_values(model->transformer_weights[index_layer]->attention_weights->weight_value[index_head][d_G], scale);
      }
      f_matrix_scale_values(model->transformer_weights[index_layer]->attention_weights->weight_output[d_G], scale);
      f_matrix_scale_values(model->transformer_weights[index_layer]->feedforward_weights->weight_hidden[d_G], scale);
      f_matrix_scale_values(model->transformer_weights[index_layer]->feedforward_weights->bias_hidden[d_G], scale);
      f_matrix_scale_values(model->transformer_weights[index_layer]->feedforward_weights->weight_output[d_G], scale);
      f_matrix_scale_values(model->transformer_weights[index_layer]->feedforward_weights->bias_output[d_G], scale);
    }
  }
  return global_gradient_normal;
}
/* keep in mind that momentum_decay tells us how much we should keep of the old gradient (e.g. 0.9 means that 90% is old momentum and 10% is the new one).
 * Same thing for the magnitude_decay and weight_decay. The epsilon could have been a constant as basically is a suuuuuper small value I'll use to prevent a
 * division by zero. Finally, step, tells us the iteration number and it is only used for bias correction
 */
void f_adam_matrix_update(t_matrix *weight, t_matrix *gradient, t_matrix *moment, t_matrix *velocity, float learning_rate, float momentum_decay,
    float magnitude_decay, float epsilon, float weight_decay, float bias_correction_momentum, float bias_correction_magnitude) {
  for (size_t index_row = 0; index_row < f_matrix_rows(weight); ++index_row)
    for (size_t index_column = 0; index_column < f_matrix_columns(weight); ++index_column) {
      float gradient_value = d_matrix_getCR(gradient, index_column, index_row),
            momentum_value = (momentum_decay * d_matrix_getCR(moment, index_column, index_row)) + ((1.0f - momentum_decay) * gradient_value),
            magnitude_value = (magnitude_decay * d_matrix_getCR(velocity, index_column, index_row)) +
          ((1.0f - magnitude_decay) * (gradient_value * gradient_value));
      d_matrix_getCR(moment, index_column, index_row) = momentum_value;
      d_matrix_getCR(velocity, index_column, index_row) = magnitude_value;
      d_matrix_getCR(weight, index_column, index_row) -= learning_rate * (momentum_value / bias_correction_momentum) /
          (sqrtf(magnitude_value / bias_correction_magnitude) + epsilon);
      d_matrix_getCR(weight, index_column, index_row) -= learning_rate * weight_decay * d_matrix_getCR(weight, index_column, index_row);
    }
}
void f_adam_weight_update(s_GPT_model *model, s_GPT_optimizer_state *optimizer_state, float learning_rate, float momentum_decay, float magnitude_decay,
    float epsilon, float weight_decay, size_t step) {
  float bias_correction_momentum = 1.0f - powf(momentum_decay, (float) step);
  float bias_correction_magnitude = 1.0f - powf(magnitude_decay, (float) step);
  f_adam_matrix_update(model->embedding_table[d_W], model->embedding_table[d_G], optimizer_state->embedding_table[d_M], optimizer_state->embedding_table[d_V],
      learning_rate, momentum_decay, magnitude_decay, epsilon, weight_decay, bias_correction_momentum, bias_correction_magnitude);
  f_adam_matrix_update(model->new_iteration_head[d_W], model->new_iteration_head[d_G], optimizer_state->new_iteration_head[d_M],
      optimizer_state->new_iteration_head[d_V], learning_rate, momentum_decay, magnitude_decay, epsilon, weight_decay, bias_correction_momentum,
      bias_correction_magnitude);
  f_adam_matrix_update(model->final_normalization_weights->gamma_weights[d_W], model->final_normalization_weights->gamma_weights[d_G],
      optimizer_state->final_normalization_state->gamma_weights[d_M], optimizer_state->final_normalization_state->gamma_weights[d_V], learning_rate,
      momentum_decay, magnitude_decay, epsilon, weight_decay, bias_correction_momentum, bias_correction_magnitude);
  f_adam_matrix_update(model->final_normalization_weights->beta_weights[d_W], model->final_normalization_weights->beta_weights[d_G],
      optimizer_state->final_normalization_state->beta_weights[d_M], optimizer_state->final_normalization_state->beta_weights[d_V], learning_rate,
      momentum_decay, magnitude_decay, epsilon, weight_decay, bias_correction_momentum, bias_correction_magnitude);
  for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer) {
    f_adam_matrix_update(model->transformer_weights[index_layer]->before_attention_weights->gamma_weights[d_W],
        model->transformer_weights[index_layer]->before_attention_weights->gamma_weights[d_G],
        optimizer_state->transformer_state[index_layer]->before_attention_state->gamma_weights[d_M],
        optimizer_state->transformer_state[index_layer]->before_attention_state->gamma_weights[d_V], learning_rate, momentum_decay, magnitude_decay, epsilon,
        weight_decay, bias_correction_momentum, bias_correction_magnitude);
    f_adam_matrix_update(model->transformer_weights[index_layer]->before_attention_weights->beta_weights[d_W],
        model->transformer_weights[index_layer]->before_attention_weights->beta_weights[d_G],
        optimizer_state->transformer_state[index_layer]->before_attention_state->beta_weights[d_M],
        optimizer_state->transformer_state[index_layer]->before_attention_state->beta_weights[d_V], learning_rate, momentum_decay, magnitude_decay, epsilon,
        weight_decay, bias_correction_momentum, bias_correction_magnitude);
    f_adam_matrix_update(model->transformer_weights[index_layer]->before_feedforward_weights->gamma_weights[d_W],
        model->transformer_weights[index_layer]->before_feedforward_weights->gamma_weights[d_G],
        optimizer_state->transformer_state[index_layer]->before_feedforward_state->gamma_weights[d_M],
        optimizer_state->transformer_state[index_layer]->before_feedforward_state->gamma_weights[d_V], learning_rate, momentum_decay, magnitude_decay, epsilon,
        weight_decay, bias_correction_momentum, bias_correction_magnitude);
    f_adam_matrix_update(model->transformer_weights[index_layer]->before_feedforward_weights->beta_weights[d_W],
        model->transformer_weights[index_layer]->before_feedforward_weights->beta_weights[d_G],
        optimizer_state->transformer_state[index_layer]->before_feedforward_state->beta_weights[d_M],
        optimizer_state->transformer_state[index_layer]->before_feedforward_state->beta_weights[d_V], learning_rate, momentum_decay, magnitude_decay, epsilon,
        weight_decay, bias_correction_momentum, bias_correction_magnitude);
    for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
      f_adam_matrix_update(model->transformer_weights[index_layer]->attention_weights->weight_query[index_head][d_W],
          model->transformer_weights[index_layer]->attention_weights->weight_query[index_head][d_G],
          optimizer_state->transformer_state[index_layer]->attention_state->weight_query[index_head][d_M],
          optimizer_state->transformer_state[index_layer]->attention_state->weight_query[index_head][d_V], learning_rate, momentum_decay, magnitude_decay,
          epsilon, weight_decay, bias_correction_momentum, bias_correction_magnitude);
      f_adam_matrix_update(model->transformer_weights[index_layer]->attention_weights->weight_key[index_head][d_W],
          model->transformer_weights[index_layer]->attention_weights->weight_key[index_head][d_G],
          optimizer_state->transformer_state[index_layer]->attention_state->weight_key[index_head][d_M],
          optimizer_state->transformer_state[index_layer]->attention_state->weight_key[index_head][d_V], learning_rate, momentum_decay, magnitude_decay,
          epsilon, weight_decay, bias_correction_momentum, bias_correction_magnitude);
      f_adam_matrix_update(model->transformer_weights[index_layer]->attention_weights->weight_value[index_head][d_W],
          model->transformer_weights[index_layer]->attention_weights->weight_value[index_head][d_G],
          optimizer_state->transformer_state[index_layer]->attention_state->weight_value[index_head][d_M],
          optimizer_state->transformer_state[index_layer]->attention_state->weight_value[index_head][d_V], learning_rate, momentum_decay, magnitude_decay,
          epsilon, weight_decay, bias_correction_momentum, bias_correction_magnitude);
    }
    f_adam_matrix_update(model->transformer_weights[index_layer]->attention_weights->weight_output[d_W],
        model->transformer_weights[index_layer]->attention_weights->weight_output[d_G],
        optimizer_state->transformer_state[index_layer]->attention_state->weight_output[d_M],
        optimizer_state->transformer_state[index_layer]->attention_state->weight_output[d_V], learning_rate, momentum_decay, magnitude_decay, epsilon,
        weight_decay, bias_correction_momentum, bias_correction_magnitude);
    f_adam_matrix_update(model->transformer_weights[index_layer]->feedforward_weights->weight_hidden[d_W],
        model->transformer_weights[index_layer]->feedforward_weights->weight_hidden[d_G],
        optimizer_state->transformer_state[index_layer]->feedforward_state->weight_hidden[d_M],
        optimizer_state->transformer_state[index_layer]->feedforward_state->weight_hidden[d_V], learning_rate, momentum_decay, magnitude_decay, epsilon,
        weight_decay, bias_correction_momentum, bias_correction_magnitude);
    f_adam_matrix_update(model->transformer_weights[index_layer]->feedforward_weights->bias_hidden[d_W],
        model->transformer_weights[index_layer]->feedforward_weights->bias_hidden[d_G],
        optimizer_state->transformer_state[index_layer]->feedforward_state->bias_hidden[d_M],
        optimizer_state->transformer_state[index_layer]->feedforward_state->bias_hidden[d_V], learning_rate, momentum_decay, magnitude_decay, epsilon,
        weight_decay, bias_correction_momentum, bias_correction_magnitude);
    f_adam_matrix_update(model->transformer_weights[index_layer]->feedforward_weights->weight_output[d_W],
        model->transformer_weights[index_layer]->feedforward_weights->weight_output[d_G],
        optimizer_state->transformer_state[index_layer]->feedforward_state->weight_output[d_M],
        optimizer_state->transformer_state[index_layer]->feedforward_state->weight_output[d_V], learning_rate, momentum_decay, magnitude_decay, epsilon,
        weight_decay, bias_correction_momentum, bias_correction_magnitude);
    f_adam_matrix_update(model->transformer_weights[index_layer]->feedforward_weights->bias_output[d_W],
        model->transformer_weights[index_layer]->feedforward_weights->bias_output[d_G],
        optimizer_state->transformer_state[index_layer]->feedforward_state->bias_output[d_M],
        optimizer_state->transformer_state[index_layer]->feedforward_state->bias_output[d_V], learning_rate, momentum_decay, magnitude_decay, epsilon,
        weight_decay, bias_correction_momentum, bias_correction_magnitude);
  }
}
void f_GPT_model_backward_new(s_GPT_model *model, t_matrix *model_forwarded_matrix /* the output of the f_GPT_model_forward_new(), the logits matrix */,
    const int *tokens, const int *target, const bool *masks, size_t length) {
  f_cross_entropy_loss_backward(model_forwarded_matrix, target, masks, length);
  t_matrix *result = f_GPT_model_head_backward_new(model, model_forwarded_matrix, model->final_normalized_sequence);
  if (result) {
    t_matrix *layer_gradient = f_layer_normalization_backward_new(model->final_normalization_weights, result,
        model->internal_states[d_number_layers - 1].output);
    for (int index_layer = (int) (d_number_layers - 1); (layer_gradient) && (index_layer >= 0); --index_layer) {
      t_matrix *input_to_layer = (index_layer > 0) ? model->internal_states[index_layer - 1].output : model->embedded_sequence;
      t_matrix *previous_gradient = f_transformer_block_backward_new(&model->internal_states[index_layer], model->transformer_weights[index_layer],
          input_to_layer, layer_gradient);
      f_matrix_free(layer_gradient);
      layer_gradient = previous_gradient;
    }
    if (layer_gradient) {
      for (size_t index_row = 0; index_row < length; ++index_row)
        for (size_t index_column = 0; index_column < d_model; ++index_column)
          d_matrix_getCR(model->embedding_table[d_G], index_column, tokens[index_row]) += d_matrix_getCR(layer_gradient, index_column, index_row);
      f_matrix_free(layer_gradient);
    }
    f_matrix_free(result);
  }
}
/* Oh JC we're almost there! Now we can implement the full backward
 * to be removed: this function verifies if the embedded table matches with the
 * requirements: a sort-of unit test */
bool p_embedded_sequence_verify(t_matrix *embedding_table, t_matrix *positional_encoding_table, t_matrix *embedded_sequence_table, const int *tokens,
    size_t tokens_length) {
  bool result = true;
  for (size_t index_sequence = 0; (index_sequence < tokens_length) && (result); ++index_sequence)
    for (size_t index_column = 0; (index_column < d_model) && (result); ++index_column)
      if (d_matrix_getCR(embedded_sequence_table, index_column, index_sequence) !=
          (d_matrix_getCR(positional_encoding_table, index_column, index_sequence) + d_matrix_getCR(embedding_table, index_column, tokens[index_sequence])))
        result = false;
  return result;
}
/* It is now time for some optimization. In this case we'll use KV (key value) cache, that stores some computation already performed
 * to reduce the number of operations we're doing. During the forward pass, in the attention step, I'm multiplying the key and value weights
 * for the token embedded. This of course, might be quite useless if you imagine the follow situation:
 *
 * Step 1: forward("hello")             - 5 tokens processed
 * Step 2: forward("hello ")            - 6 tokens processed
 * Step 3: forward("hello t")           - 7 tokens processed
 * Step 4: forward("hello th")          - 8 tokens processed
 * {...}
 * Step N: forward("hello the l ...")   - N tokens processed
 *
 * As we're doing key[i] = weight_key x embed(token[i]) and value[i] = value x embed(token[i]) it is kind of useless to repeat the operation
 * for all the tokens except for the last one.
 * */
typedef struct s_kv_cache {
  t_matrix *key[d_number_heads], *value[d_number_heads];
  size_t length;
} s_kv_cache;
typedef struct s_GPT_kv_cache {
  s_kv_cache cache[d_number_layers];
} s_GPT_kv_cache;
void f_GPT_kv_cache_free(s_GPT_kv_cache *cache) {
  if (cache) {
    for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer)
      for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
        f_matrix_free(cache->cache[index_layer].key[index_head]);
        f_matrix_free(cache->cache[index_layer].value[index_head]);
      }
    free(cache);
  }
}
s_GPT_kv_cache *f_GPT_kv_cache_new(void) {
  s_GPT_kv_cache *result = (s_GPT_kv_cache *) malloc(sizeof(s_GPT_kv_cache));
  if (result) {
    memset(result, 0, sizeof(s_GPT_kv_cache));
    for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer)
      for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
        result->cache[index_layer].key[index_head] = f_matrix_new(d_context, d_size_head);
        result->cache[index_layer].value[index_head] = f_matrix_new(d_context, d_size_head);
        if ((!result->cache[index_layer].key[index_head]) || (!result->cache[index_layer].value[index_head])) {
          f_GPT_kv_cache_free(result);
          return NULL;
        }
      }
  }
  return result;
}
void f_GPT_kv_cache_zero(s_GPT_kv_cache *cache) {
  if (cache)
    for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer)
      cache->cache[index_layer].length = 0;
}
t_matrix *f_attention_decode_step(s_kv_cache *cache, s_attention_weights *weights, t_matrix *embedded_sequence) {
  d_assert(cache->length < d_context);
  t_matrix *result = NULL, *collection = f_matrix_new(1, (d_number_heads * d_size_head));
  if (collection) {
    const float sqrt_head = sqrtf((float) d_size_head);
    t_matrix *query = NULL, *key = NULL, *value = NULL;
    float scores[d_context];
    size_t total_length = cache->length + 1;
    for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
      if (((query = f_matrix_multiply(query, embedded_sequence, weights->weight_query[index_head][d_W]))) &&
          ((key = f_matrix_multiply(key, embedded_sequence, weights->weight_key[index_head][d_W]))) &&
          ((value = f_matrix_multiply(value, embedded_sequence, weights->weight_value[index_head][d_W])))) {
        /* append k and v into the cache at row cache->length */
        for (size_t index_size = 0; index_size < d_size_head; ++index_size) {
          d_matrix_getCR(cache->key[index_head], index_size, cache->length) = d_matrix_getCR(key, index_size, 0);
          d_matrix_getCR(cache->value[index_head], index_size, cache->length) = d_matrix_getCR(value, index_size, 0);
        }
        for (size_t index_token = 0; index_token < total_length; ++index_token) {
          scores[index_token] = 0;
          for (size_t index_size = 0; index_size < d_size_head; ++index_size)
            scores[index_token] += d_matrix_getCR(query, index_size, 0) * d_matrix_getCR(cache->key[index_head], index_size, index_token);
          scores[index_token] /= sqrt_head;
        }
        f_vector_softmax(scores, total_length);
        /* weighted sum of value rows are getting written into the collection at columns [h*d_size_head .. (h+1)*d_size_head) */
        for (size_t index_size = 0; index_size < d_size_head; ++index_size) {
          float output_value = 0;
          for (size_t index_token = 0; index_token < total_length; ++index_token)
            output_value += scores[index_token] * d_matrix_getCR(cache->value[index_head], index_size, index_token);
          d_matrix_getCR(collection, d_size_head * index_head + index_size, 0) = output_value;
        }
      }
    }
    f_matrix_free(query);
    f_matrix_free(key);
    f_matrix_free(value);
    result = f_matrix_multiply(NULL, collection, weights->weight_output[d_W]);
    f_matrix_free(collection);
    ++(cache->length);
  }
  return result;
}
t_matrix *f_transformer_decode_step(s_kv_cache *cache, s_transformer_weights *weights, t_matrix *embedded_sequence) {
  t_matrix *result = NULL, *attention_normalized = f_layer_normalization_forward_new(weights->before_attention_weights, embedded_sequence);
  if (attention_normalized) {
    t_matrix *attention_residual = f_attention_decode_step(cache, weights->attention_weights, attention_normalized);
    if (attention_residual) {
      t_matrix *attention_residual_normalized;
      f_matrix_add_in_place(attention_residual, embedded_sequence); /* residual: attention_residual += embedded tokens */
      if ((attention_residual_normalized = f_layer_normalization_forward_new(weights->before_feedforward_weights, attention_residual))) {
        if ((result = f_feedforward_encoded_output_new(NULL, weights->feedforward_weights, attention_residual_normalized)))
          f_matrix_add_in_place(result, attention_residual); /* residual: result += attention_residual */
        f_matrix_free(attention_residual_normalized);
      }
      f_matrix_free(attention_residual);
    }
    f_matrix_free(attention_normalized);
  }
  return result;
}
void f_checkpoint_save(s_GPT_model *model, s_GPT_optimizer_state *state, size_t step, const char *path) {
  FILE *stream;
  char temporary_file_path[PATH_MAX];
  /* temporary file first, then renamed */
  snprintf(temporary_file_path, (PATH_MAX - 1), "%s.tmp", path);
  if ((stream = fopen(temporary_file_path, "w"))) {
    fprintf(stream, "%x %zu ", 0xdeadbeef, step);
    /* model weights */
    f_matrix_write(stream, model->embedding_table[d_W]);
    f_matrix_write(stream, model->positional_encoding_table);
    f_matrix_write(stream, model->new_iteration_head[d_W]);
    for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer) {
      f_matrix_write(stream, model->transformer_weights[index_layer]->before_attention_weights->gamma_weights[d_W]);
      f_matrix_write(stream, model->transformer_weights[index_layer]->before_attention_weights->beta_weights[d_W]);
      f_matrix_write(stream, model->transformer_weights[index_layer]->before_feedforward_weights->gamma_weights[d_W]);
      f_matrix_write(stream, model->transformer_weights[index_layer]->before_feedforward_weights->beta_weights[d_W]);
      for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
        f_matrix_write(stream, model->transformer_weights[index_layer]->attention_weights->weight_query[index_head][d_W]);
        f_matrix_write(stream, model->transformer_weights[index_layer]->attention_weights->weight_key[index_head][d_W]);
        f_matrix_write(stream, model->transformer_weights[index_layer]->attention_weights->weight_value[index_head][d_W]);
      }
      f_matrix_write(stream, model->transformer_weights[index_layer]->attention_weights->weight_output[d_W]);
      f_matrix_write(stream, model->transformer_weights[index_layer]->feedforward_weights->weight_hidden[d_W]);
      f_matrix_write(stream, model->transformer_weights[index_layer]->feedforward_weights->bias_hidden[d_W]);
      f_matrix_write(stream, model->transformer_weights[index_layer]->feedforward_weights->weight_output[d_W]);
      f_matrix_write(stream, model->transformer_weights[index_layer]->feedforward_weights->bias_output[d_W]);
    }
    f_matrix_write(stream, model->final_normalization_weights->gamma_weights[d_W]);
    f_matrix_write(stream, model->final_normalization_weights->beta_weights[d_W]);
    /* optimizer state — M then V for every matrix, same order */
    for (size_t index_moment = d_M; index_moment <= d_V; ++index_moment) {
      f_matrix_write(stream, state->embedding_table[index_moment]);
      f_matrix_write(stream, state->new_iteration_head[index_moment]);
      for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer) {
        f_matrix_write(stream, state->transformer_state[index_layer]->before_attention_state->gamma_weights[index_moment]);
        f_matrix_write(stream, state->transformer_state[index_layer]->before_attention_state->beta_weights[index_moment]);
        f_matrix_write(stream, state->transformer_state[index_layer]->before_feedforward_state->gamma_weights[index_moment]);
        f_matrix_write(stream, state->transformer_state[index_layer]->before_feedforward_state->beta_weights[index_moment]);
        for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
          f_matrix_write(stream, state->transformer_state[index_layer]->attention_state->weight_query[index_head][index_moment]);
          f_matrix_write(stream, state->transformer_state[index_layer]->attention_state->weight_key[index_head][index_moment]);
          f_matrix_write(stream, state->transformer_state[index_layer]->attention_state->weight_value[index_head][index_moment]);
        }
        f_matrix_write(stream, state->transformer_state[index_layer]->attention_state->weight_output[index_moment]);
        f_matrix_write(stream, state->transformer_state[index_layer]->feedforward_state->weight_hidden[index_moment]);
        f_matrix_write(stream, state->transformer_state[index_layer]->feedforward_state->bias_hidden[index_moment]);
        f_matrix_write(stream, state->transformer_state[index_layer]->feedforward_state->weight_output[index_moment]);
        f_matrix_write(stream, state->transformer_state[index_layer]->feedforward_state->bias_output[index_moment]);
      }
      f_matrix_write(stream, state->final_normalization_state->gamma_weights[index_moment]);
      f_matrix_write(stream, state->final_normalization_state->beta_weights[index_moment]);
    }
    fclose(stream);
    rename(temporary_file_path, path);
  }
}
/* pay attention: if state is NULL, we'll not going to be loading the optimizer (e.g. during SFT training) */
int f_checkpoint_load(s_GPT_model *model, s_GPT_optimizer_state *state, size_t *step, const char *path) {
  int result = 0;
  unsigned int magic;
  FILE *stream;
  if ((stream = fopen(path, "r"))) {
    size_t step_reached;
    fscanf(stream, "%x %zu ", &magic, &step_reached);
    if (magic == 0xdeadbeef) {
      result = 1;
      if (step)
        *step = step_reached;
      /* model weights */
      f_matrix_read(stream, model->embedding_table[d_W]);
      f_matrix_read(stream, model->positional_encoding_table);
      f_matrix_read(stream, model->new_iteration_head[d_W]);
      for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer) {
        f_matrix_read(stream, model->transformer_weights[index_layer]->before_attention_weights->gamma_weights[d_W]);
        f_matrix_read(stream, model->transformer_weights[index_layer]->before_attention_weights->beta_weights[d_W]);
        f_matrix_read(stream, model->transformer_weights[index_layer]->before_feedforward_weights->gamma_weights[d_W]);
        f_matrix_read(stream, model->transformer_weights[index_layer]->before_feedforward_weights->beta_weights[d_W]);
        for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
          f_matrix_read(stream, model->transformer_weights[index_layer]->attention_weights->weight_query[index_head][d_W]);
          f_matrix_read(stream, model->transformer_weights[index_layer]->attention_weights->weight_key[index_head][d_W]);
          f_matrix_read(stream, model->transformer_weights[index_layer]->attention_weights->weight_value[index_head][d_W]);
        }
        f_matrix_read(stream, model->transformer_weights[index_layer]->attention_weights->weight_output[d_W]);
        f_matrix_read(stream, model->transformer_weights[index_layer]->feedforward_weights->weight_hidden[d_W]);
        f_matrix_read(stream, model->transformer_weights[index_layer]->feedforward_weights->bias_hidden[d_W]);
        f_matrix_read(stream, model->transformer_weights[index_layer]->feedforward_weights->weight_output[d_W]);
        f_matrix_read(stream, model->transformer_weights[index_layer]->feedforward_weights->bias_output[d_W]);
      }
      f_matrix_read(stream, model->final_normalization_weights->gamma_weights[d_W]);
      f_matrix_read(stream, model->final_normalization_weights->beta_weights[d_W]);
      /* optimizer state (only if exists) */
      if (state)
        for (size_t index_moment = d_M; index_moment <= d_V; ++index_moment) {
          f_matrix_read(stream, state->embedding_table[index_moment]);
          f_matrix_read(stream, state->new_iteration_head[index_moment]);
          for (size_t index_layer = 0; index_layer < d_number_layers; ++index_layer) {
            f_matrix_read(stream, state->transformer_state[index_layer]->before_attention_state->gamma_weights[index_moment]);
            f_matrix_read(stream, state->transformer_state[index_layer]->before_attention_state->beta_weights[index_moment]);
            f_matrix_read(stream, state->transformer_state[index_layer]->before_feedforward_state->gamma_weights[index_moment]);
            f_matrix_read(stream, state->transformer_state[index_layer]->before_feedforward_state->beta_weights[index_moment]);
            for (size_t index_head = 0; index_head < d_number_heads; ++index_head) {
              f_matrix_read(stream, state->transformer_state[index_layer]->attention_state->weight_query[index_head][index_moment]);
              f_matrix_read(stream, state->transformer_state[index_layer]->attention_state->weight_key[index_head][index_moment]);
              f_matrix_read(stream, state->transformer_state[index_layer]->attention_state->weight_value[index_head][index_moment]);
            }
            f_matrix_read(stream, state->transformer_state[index_layer]->attention_state->weight_output[index_moment]);
            f_matrix_read(stream, state->transformer_state[index_layer]->feedforward_state->weight_hidden[index_moment]);
            f_matrix_read(stream, state->transformer_state[index_layer]->feedforward_state->bias_hidden[index_moment]);
            f_matrix_read(stream, state->transformer_state[index_layer]->feedforward_state->weight_output[index_moment]);
            f_matrix_read(stream, state->transformer_state[index_layer]->feedforward_state->bias_output[index_moment]);
          }
          f_matrix_read(stream, state->final_normalization_state->gamma_weights[index_moment]);
          f_matrix_read(stream, state->final_normalization_state->beta_weights[index_moment]);
        }
    }
    fclose(stream);
  }
  return result;
}
#define d_supervised_fine_tuning_prefix_assistant 'A'
#define d_supervised_fine_tuning_prefix_system 'S'
#define d_supervised_fine_tuning_prefix_user 'U'
size_t f_supervised_fine_tuning_load(const char *training_module, int **tokens, bool **masks) {
  FILE *corpus_stream;
  size_t result = 0;
  if ((corpus_stream = fopen(training_module, "r"))) {
    size_t corpus_length = 0;
    fseek(corpus_stream, 0, SEEK_END);
    corpus_length = (size_t) ftell(corpus_stream);
    fseek(corpus_stream, 0, SEEK_SET);
    if (((*tokens) = (int *) malloc(sizeof(int) * corpus_length)) && ((*masks) = (bool *) malloc(sizeof(bool) * corpus_length))) {
      char *corpus_payload = NULL;
      if ((corpus_payload = (char *) malloc(corpus_length + 1))) {
        bool end_of_sequence = true;
        char *start_block = corpus_payload, *end_block = NULL;
        fread(corpus_payload, 1, corpus_length, corpus_stream);
        corpus_payload[corpus_length] = 0;
        while ((*start_block != 0) && (((end_block = strchr(start_block, '\n'))) || ((end_block = (start_block + strlen(start_block)))))) {
          size_t length_block = (end_block - start_block) + 1;
          if (length_block > 2) {
            bool mask_active = false;
            if (start_block[1] == ':') {
              size_t written_tokens;
              if (end_of_sequence) {
                end_of_sequence = false;
                (*masks)[result] = 0;
                (*tokens)[result++] = d_token_bos;
              }
              switch (start_block[0]) {
                case d_supervised_fine_tuning_prefix_assistant: {
                  (*masks)[result] = 0;
                  (*tokens)[result++] = d_token_ast;
                  mask_active = true;
                  break;
                }
                case d_supervised_fine_tuning_prefix_system: {
                  (*masks)[result] = 0;
                  (*tokens)[result++] = d_token_sys;
                  break;
                }
                default: /* unrecognizable token, we're going to fall back to the user */
                case d_supervised_fine_tuning_prefix_user: {
                  (*masks)[result] = 0;
                  (*tokens)[result++] = d_token_usr;
                  break;
                }
              }
              written_tokens = f_encode((start_block + 2), &((*tokens)[result]), (length_block - 2));
              memset(&((*masks)[result]), (mask_active) ? 1 : 0, (sizeof(bool) * written_tokens));
              result += written_tokens;
            }
          } else {
            /* we want to avoid multiple d_token_eos to be one after the other */
            if ((result > 0) && ((*tokens)[result - 1] != d_token_eos)) {
              (*masks)[result] = 1;
              (*tokens)[result++] = d_token_eos;
            }
            end_of_sequence = true;
          }
          start_block = end_block;
          if (*start_block == '\n') /* jump to the next token */
            ++start_block;
        }
        free(corpus_payload);
      }
    }
    if (!result) {
      if (*tokens)
        free(*tokens);
      if (*masks)
        free(*masks);
      *tokens = NULL;
      *masks = NULL;
    }
    fclose(corpus_stream);
  }
  return result;
}
size_t f_get_best_token(t_matrix *logits, int *tokens, size_t length) {
  float best_value = d_matrix_getCR(logits, 0, (length - 1));
  size_t best_token_index = 0;
  for (size_t index_token = 1; index_token < d_vocabulary_size; ++index_token) {
    float current_value = d_matrix_getCR(logits, index_token, (length - 1));
    if (current_value > best_value) {
      best_value = current_value;
      best_token_index = index_token;
    }
  }
  return best_token_index;
}
size_t f_sample_token(t_matrix *logits, size_t position, float temperature, size_t top_k, float top_p) {
  float local[d_vocabulary_size];
  size_t result;
  for (size_t index_token = 0; index_token < d_vocabulary_size; ++index_token)
    local[index_token] = d_matrix_getCR(logits, index_token, position);
  if (temperature <= 0.0f) {
    result = 0;
    for (size_t index_token = 1; index_token < d_vocabulary_size; ++index_token)
      if (local[index_token] > local[result])
        result = index_token;
  } else {
    float random_value, cumulative;
    for (size_t index_token = 0; index_token < d_vocabulary_size; ++index_token)
      local[index_token] /= temperature;
    if ((top_k > 0) && (top_k < d_vocabulary_size)) {
      float sorted[d_vocabulary_size], threshold, tmp;
      size_t max_index;
      memcpy(sorted, local, sizeof(sorted));
      /* sorting (maybe I can avoid this with a bitwise mask array?) */
      for (size_t index_k = 0; index_k < top_k; ++index_k) {
        max_index = index_k;
        for (size_t index_j = index_k + 1; index_j < d_vocabulary_size; ++index_j)
          if (sorted[index_j] > sorted[max_index])
            max_index = index_j;
        /* swap */
        tmp = sorted[index_k];
        sorted[index_k] = sorted[max_index];
        sorted[max_index] = tmp;
      }
      threshold = sorted[top_k - 1];
      for (size_t index_token = 0; index_token < d_vocabulary_size; ++index_token)
        if (local[index_token] < threshold)
          local[index_token] = -1e9f; /* BAAAAAM, you're dead now */
    }
    f_vector_softmax(local, d_vocabulary_size);
    if ((top_p > 0.0f) && (top_p < 1.0f)) {
      size_t order[d_vocabulary_size], key, cutoff, index_j;
      float total;
      for (size_t index_token = 0; index_token < d_vocabulary_size; ++index_token)
        order[index_token] = index_token;
      for (size_t index_i = 1; index_i < d_vocabulary_size; ++index_i) {
        key = order[index_i];
        index_j = index_i;
        while ((index_j > 0) && (local[order[index_j - 1]] < local[key])) {
          order[index_j] = order[index_j - 1];
          --index_j;
        }
        order[index_j] = key;
      }
      cumulative = 0.0f;
      cutoff = d_vocabulary_size;
      for (size_t index_token = 0; index_token < d_vocabulary_size; ++index_token) {
        cumulative += local[order[index_token]];
        if (cumulative >= top_p) {
          cutoff = index_token + 1;
          break;
        }
      }
      for (size_t index_token = cutoff; index_token < d_vocabulary_size; ++index_token)
        local[order[index_token]] = 0.0f;
      total = 0.0f;
      for (size_t index_token = 0; index_token < d_vocabulary_size; ++index_token)
        total += local[index_token];
      if (total > 0.0f)
        for (size_t index_token = 0; index_token < d_vocabulary_size; ++index_token)
          local[index_token] /= total;
    }
    random_value = (float) rand() / ((float) RAND_MAX + 1.0f);
    cumulative = 0.0f;
    result = d_vocabulary_size - 1;
    for (size_t index_token = 0; index_token < d_vocabulary_size; ++index_token) {
      cumulative += local[index_token];
      if (random_value < cumulative) {
        result = index_token;
        break;
      }
    }
  }
  return result;
}
float f_token_accuracy_percentage(t_matrix *logits, const int *targets, size_t length) {
  size_t matches = length;
  for (size_t index_target = 0; index_target < length; ++index_target) {
    float max_value = d_matrix_getCR(logits, targets[index_target], index_target);
    for (size_t index_column = 0; index_column < f_matrix_columns(logits); ++index_column)
      if (d_matrix_getCR(logits, index_column, index_target) > max_value) {
        --matches;
        break; /* it is a miss! We would have picked something different */
      }
  }
  return ((float) matches / (float) length) * 100.0f;
}
#define d_top_k 40 /* only the top-k tokens by logit survive before sampling; 0 = keep all */
#define d_top_p 0.9f /* keep the smallest set of tokens whose cumulative probability >= p; 1.0 = no cutoff */
#define d_default_temperature 0.8f /* chat sampling temperature default; lower = more focused, higher = more creative */
#define d_learning_rate 3e-4f /* size of each weight update step; too high = unstable training, too low = slow convergence */
#define d_momentum_decay 0.9f /* how much of the previous gradient direction carries into the current step (Adam β1) */
#define d_magnitude_decay 0.999f /* how much of the previous gradient magnitude history carries into the current step (Adam β2) */
#define d_weight_decay 0.01f /* gently shrinks weights each step to discourage memorisation (L2 regularisation) */
#define d_epsilon 1e-8f /* tiny floor added to the denominator to avoid division by zero when gradient magnitude is near zero */
void f_run_pretraining(const char *corpus_path, const char *model_path, const size_t number_epochs) {
  FILE *corpus_stream;
  if ((corpus_stream = fopen(corpus_path, "r"))) {
    size_t corpus_length = 0, total_steps_per_epoch = 0;
    char *corpus_payload = NULL;
    fseek(corpus_stream, 0, SEEK_END);
    corpus_length = (size_t) ftell(corpus_stream);
    fseek(corpus_stream, 0, SEEK_SET);
    total_steps_per_epoch = (corpus_length / d_context);
    if ((corpus_payload = (char *) malloc(corpus_length + 1))) { /* bleargh, everything goes in memory. Quite annoying, right? */
      s_GPT_model *model = f_GPT_model_new(d_context);
      s_GPT_optimizer_state *optimizer_state = f_GPT_optimizer_state_new();
      t_matrix *model_forward = NULL;
      size_t step = 1, residual_characters, starting_epoch = 0, starting_step = 0, starting_corpus_offset = 0;
      int input_tokens[d_context], target_tokens[d_context];
      fread(corpus_payload, 1, corpus_length, corpus_stream);
      corpus_payload[corpus_length] = '\0';
      if (f_checkpoint_load(model, optimizer_state, &step, model_path)) {
        starting_epoch = ((step - 1) / total_steps_per_epoch);
        starting_step = ((step - 1) % total_steps_per_epoch);
        starting_corpus_offset = starting_step * d_context;
        printf("resuming from checkpoint at step %zu (epoch %zu, step %zu)\n", step, (starting_epoch + 1), starting_step);
      }
      for (size_t index_epoch = starting_epoch; index_epoch < number_epochs; ++index_epoch) {
        for (size_t index_corpus = starting_corpus_offset; (index_corpus < corpus_length) && ((residual_characters = (corpus_length - index_corpus)) > 2);
            index_corpus += d_context) {
          size_t chunk_size = d_context;
          if (residual_characters < (d_context + 1))
            chunk_size = (residual_characters - 1);
          f_encode(&(corpus_payload[index_corpus]), input_tokens, chunk_size);
          f_encode(&(corpus_payload[index_corpus + 1]), target_tokens, chunk_size);
          f_GPT_model_gradient_zero(model);
          if ((model_forward = f_GPT_model_forward_new(model, input_tokens, chunk_size))) {
            float global_gradient_normal, loss = f_cross_entropy_loss(model_forward, target_tokens, NULL, chunk_size),
                                          accuracy = f_token_accuracy_percentage(model_forward, target_tokens, chunk_size);
            f_GPT_model_backward_new(model, model_forward, input_tokens, target_tokens, NULL, chunk_size);
            global_gradient_normal = f_gradients_clip(model, 1.0);
            printf("Epoch %zu/%zu (step %zu, per epoch %zu) | metrics: bits-per-character (BPC) %.01f | loss %.03f | perplexity %.02f | accuracy %.02f%% | "
                   "gradient normal (training stability) %.02f\n",
                (index_epoch + 1), number_epochs, step, total_steps_per_epoch,
                /* bits per character */ (loss / logf(2.0)),
                /* raw loss */ loss,
                /* perplexity */ (expf(loss)),
                /* accuracy */ accuracy,
                /* gradient normal, tells the stability of the learning (near zero, we're not learning anymore) */ global_gradient_normal);
            f_adam_weight_update(model, optimizer_state, d_learning_rate, d_momentum_decay, d_magnitude_decay, d_epsilon, d_weight_decay, step);
            f_checkpoint_save(model, optimizer_state, step, model_path);
            f_matrix_free(model_forward);
            model_forward = NULL;
          }
          ++step;
        }
        starting_corpus_offset = 0;
      }
      f_GPT_optimizer_state_free(optimizer_state);
      f_GPT_model_free(model);
      free(corpus_payload);
    }
    fclose(corpus_stream);
  } else
    fprintf(stderr, "cannot load corpus '%s'\n", corpus_path);
}
void f_run_supervised_fine_tuning(const char *corpus_path, const char *source_model_path, const char *destination_model_path, const size_t number_epochs) {
  int *supervised_fine_tuning_tokens = NULL;
  bool *supervised_fine_tuning_masks = NULL;
  size_t supervised_fine_tuning_length = f_supervised_fine_tuning_load(corpus_path, &supervised_fine_tuning_tokens, &supervised_fine_tuning_masks);
  if (supervised_fine_tuning_length > 1) {
    s_GPT_model *model = f_GPT_model_new(d_context);
    if (f_checkpoint_load(model, NULL, NULL, source_model_path)) {
      s_GPT_optimizer_state *optimizer_state = f_GPT_optimizer_state_new();
      t_matrix *model_forward = NULL;
      int input_tokens[d_context], target_tokens[d_context];
      bool chunk_mask[d_context];
      size_t step = 1;
      printf("loaded pre-trained model from '%s'\n", source_model_path);
      for (size_t index_epoch = 0; index_epoch < number_epochs; ++index_epoch)
        for (size_t index_corpus = 0; (index_corpus + 1) < supervised_fine_tuning_length;) {
          /* find the end of this conversation: scan forward to <EOS> or end of stream */
          size_t index_conversation_end = index_corpus, chunk_size;
          while ((index_conversation_end < supervised_fine_tuning_length) && (supervised_fine_tuning_tokens[index_conversation_end] != d_token_eos))
            ++index_conversation_end;
          /* in case the conversation is empty, just d_token_eos or as single, useless token that doesn't terminate with d_token_eos, or bigger than d_context,
           * we'll skip it
           */
          if (((chunk_size = (index_conversation_end - index_corpus)) > 0) && (chunk_size < d_context)) {
            for (size_t index_chunk = 0; index_chunk < chunk_size; ++index_chunk) {
              input_tokens[index_chunk] = supervised_fine_tuning_tokens[index_corpus + index_chunk];
              target_tokens[index_chunk] = supervised_fine_tuning_tokens[index_corpus + index_chunk + 1];
              chunk_mask[index_chunk] = supervised_fine_tuning_masks[index_corpus + index_chunk + 1];
            }
            /* now we check if the conversation ends with d_token_eos, otherwise we'll force it if we have enough space */
            if (((chunk_size + 1) < d_context) && (target_tokens[(chunk_size - 1)] != d_token_eos)) {
              input_tokens[chunk_size] = target_tokens[(chunk_size - 1)];
              target_tokens[chunk_size] = d_token_eos;
              chunk_mask[chunk_size] = true; /* the EOS is always training */
              ++chunk_size;
            }
            /* if d_token_eos still doesn't exists, we're skipping the stack */
            if (target_tokens[(chunk_size - 1)] == d_token_eos) {
              f_GPT_model_gradient_zero(model);
              if ((model_forward = f_GPT_model_forward_new(model, input_tokens, chunk_size))) {
                float global_gradient_normal, loss = f_cross_entropy_loss(model_forward, target_tokens, chunk_mask, chunk_size),
                                              accuracy = f_token_accuracy_percentage(model_forward, target_tokens, chunk_size);
                f_GPT_model_backward_new(model, model_forward, input_tokens, target_tokens, chunk_mask, chunk_size);
                global_gradient_normal = f_gradients_clip(model, 1.0);
                printf("Epoch %zu/%zu (step %zu) | metrics: bits-per-character (BPC) %.01f | loss %.03f | perplexity %.02f | accuracy %.02f%% | "
                       "gradient normal (training stability) %.02f\n",
                    (index_epoch + 1), number_epochs, step,
                    /* bits per character */ (loss / logf(2.0)),
                    /* raw loss */ loss,
                    /* perplexity */ (expf(loss)),
                    /* accuracy */ accuracy,
                    /* gradient normal, tells the stability of the learning (near zero, we're not learning anymore) */ global_gradient_normal);
                f_adam_weight_update(model, optimizer_state,
                    (d_learning_rate * 0.1f /* we're learning much slower, as we don't want to break the existing model */), d_momentum_decay,
                    d_magnitude_decay, d_epsilon, d_weight_decay, step);
                f_checkpoint_save(model, optimizer_state, step, destination_model_path);
                f_matrix_free(model_forward);
              }
              ++step;
            }
          }
          index_corpus = (index_conversation_end + 1); /* advance to next conversation */
        }
      f_GPT_optimizer_state_free(optimizer_state);
    }
    f_GPT_model_free(model);
  }
  if (supervised_fine_tuning_tokens)
    free(supervised_fine_tuning_tokens);
  if (supervised_fine_tuning_masks)
    free(supervised_fine_tuning_masks);
}
int main(int argc, char *argv[]) {
  if ((argc >= 4) && (argv[1][0] == 't')) {
    size_t number_epochs = 1;
    if (argc > 4)
      number_epochs = atoi(argv[4]);
    f_run_pretraining(argv[2], argv[3], number_epochs);
  } else if ((argc >= 4) && (argv[1][0] == 'f')) {
    size_t number_epochs = 1;
    if (argc > 5)
      number_epochs = atoi(argv[5]);
    f_run_supervised_fine_tuning(argv[2], argv[3], argv[4], number_epochs);
  } else {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "  %s t <corpus> <model> [epochs]\n", argv[0]);
    fprintf(stderr, "  %s f <supervised_fine_tuning_data> <pretrained_model> <supervised_fine_tuning_model> [epochs]\n", argv[0]);
    fprintf(stderr, "  %s c <model> \"<system prompt>\" [temperature]\n", argv[0]);
  }
  return 0;
}
