// The code structure (especially file reading and saving functions) is adapted from the Word2Vec implementation
//          https://github.com/tmikolov/word2vec

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define MAX_STRING 100
#define ACOS_TABLE_SIZE 5000
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int corpus_max_size = 40000000;  // Maximum 40M documents in the corpus

typedef float real;

struct vocab_word {
    long long cn;
    char *word;
};

char train_file[MAX_STRING], load_emb_file[MAX_STRING];
char word_emb[MAX_STRING], context_emb[MAX_STRING], doc_output[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int debug_mode = 2, window = 5, min_count = 5, num_threads = 20, min_reduce = 1;
int *vocab_hash, *docs;
long long *doc_sizes;
long long vocab_max_size = 1000, vocab_size = 0, corpus_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 10, file_size = 0;
int negative = 2;
const int table_size = 1e8;
int *word_table;
real alpha = 0.04, starting_alpha, sample = 1e-3, margin = 0.15;
real *syn0, *syn1neg, *syn1doc;
clock_t start;


void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  word_table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    word_table[a] = i;
    if (a / (double) table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *) "</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Locate line number of current file pointer
int FindLine(FILE *fin) {
  long long pos = ftell(fin);
  long long lo = 0, hi = corpus_size - 1;
  while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;
    if (doc_sizes[mid] > pos) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) { // assert all sortings will be the same (since c++ qsort is not stable..)
  if (((struct vocab_word *) b)->cn == ((struct vocab_word *) a)->cn) {
    return strcmp(((struct vocab_word *) b)->word, ((struct vocab_word *) a)->word);
  }
  return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

int IntCompare(const void * a, const void * b) 
{ 
  return ( *(int*)a - *(int*)b ); 
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *) "</s>");

  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } 
    else if (i == 0) {
      vocab[i].cn++;
      doc_sizes[corpus_size] = ftell(fin);
      corpus_size++;
    }
    else {
      vocab[i].cn++;
      docs[corpus_size]++;
    }
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void LoadEmb(char *emb_file, real *emb_ptr) {
  long long a, b;
  int *vocab_match_tmp = (int *) calloc(vocab_size, sizeof(int));
  int vocab_size_tmp = 0, word_dim;
  char *current_word = (char *) calloc(MAX_STRING, sizeof(char));
  real *syn_tmp = NULL, norm;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn_tmp, 128, (long long) layer1_size * sizeof(real));
  if (syn_tmp == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  printf("Loading embedding from file %s\n", emb_file);
  if (access(emb_file, R_OK) == -1) {
    printf("File %s does not exist\n", emb_file);
    exit(1);
  }
  // read embedding file
  FILE *fp = fopen(emb_file, "r");
  fscanf(fp, "%d", &vocab_size_tmp);
  fscanf(fp, "%d", &word_dim);
  if (layer1_size != word_dim) {
    printf("Embedding dimension incompatible with pretrained file!\n");
    exit(1);
  }
  vocab_size_tmp = 0;
  while (1) {
    fscanf(fp, "%s", current_word);
    a = SearchVocab(current_word);
    if (a == -1) {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &syn_tmp[b]);
    }
    else {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &emb_ptr[a * layer1_size + b]);
      vocab_match_tmp[vocab_size_tmp] = a;
      vocab_size_tmp++;
    }
    if (feof(fp)) break;
  }
  printf("In vocab: %d\n", vocab_size_tmp);
  qsort(&vocab_match_tmp[0], vocab_size_tmp, sizeof(int), IntCompare);
  vocab_match_tmp[vocab_size_tmp] = vocab_size;
  int i = 0;
  for (a = 0; a < vocab_size; a++) {
    if (a < vocab_match_tmp[i]) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        emb_ptr[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        norm += emb_ptr[a * layer1_size + b] * emb_ptr[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        emb_ptr[a * layer1_size + b] /= sqrt(norm);
    }
    else if (i < vocab_size_tmp) {
      i++;
    }
  }
  fclose(fp);
  free(current_word);
  free(emb_file);
  free(vocab_match_tmp);
  free(syn_tmp);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }

  a = posix_memalign((void **) &syn1neg, 128, (long long) vocab_size * layer1_size * sizeof(real));
  a = posix_memalign((void **) &syn1doc, 128, (long long) corpus_size * layer1_size * sizeof(real));
  if (syn1neg == NULL) {
    printf("Memory allocation failed (syn1neg)\n");
    exit(1);
  }
  if (syn1doc == NULL) {
    printf("Memory allocation failed (syn1doc)\n");
    exit(1);
  }
  
  real norm;
  if (load_emb_file[0] != 0) {
    char *center_emb_file = (char *) calloc(MAX_STRING, sizeof(char));
    char *context_emb_file = (char *) calloc(MAX_STRING, sizeof(char));
    strcpy(center_emb_file, load_emb_file);
    strcat(center_emb_file, "_w.txt");
    strcpy(context_emb_file, load_emb_file);
    strcat(context_emb_file, "_v.txt");
    LoadEmb(center_emb_file, syn0);
    LoadEmb(context_emb_file, syn1neg);
  }
  else {
    for (a = 0; a < vocab_size; a++) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        syn1neg[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        norm += syn1neg[a * layer1_size + b] * syn1neg[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] /= sqrt(norm);
    }
    for (a = 0; a < vocab_size; a++) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        norm += syn0[a * layer1_size + b] * syn0[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        syn0[a * layer1_size + b] /= sqrt(norm);
    }
  }

  for (a = 0; a < corpus_size; a++) {
    norm = 0.0;
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long) 25214903917 + 11;
      syn1doc[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
      norm += syn1doc[a * layer1_size + b] * syn1doc[a * layer1_size + b];
    }
    for (b = 0; b < layer1_size; b++)
      syn1doc[a * layer1_size + b] /= sqrt(norm);
  }
  
}

void *TrainModelThread(void *id) {
  long long a, b, d, doc = 0, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, l3 = 0, c, target, local_iter = iter;
  unsigned long long next_random = (long long) id;
  real f, g, h, step, obj_w = 0, obj_d = 0;
  clock_t now;
  real *neu1 = (real *) calloc(layer1_size, sizeof(real));
  real *grad = (real *) calloc(layer1_size, sizeof(real));
  real *neu1e = (real *) calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);

  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Objective (w): %f  Objective (d): %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha, 
               obj_w, obj_d, word_count_actual / (real) (iter * train_words + 1) * 100,
               word_count_actual / ((real) (now - start + 1) / (real) CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      doc = FindLine(fi);
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) /
                     vocab[word].cn;
          next_random = next_random * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real) 65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);
      continue;
    }

    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long) 25214903917 + 11;
    b = next_random % window;

    for (a = b; a < window * 2 + 1 - b; a++)
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size; // positive center word u
        
        obj_w = 0;
        for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            l3 = word * layer1_size; // positive context word v
          } else {
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            target = word_table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            l2 = target * layer1_size; // negative center word u'
            f = 0;
            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l3]; // f = cos(v, u) = v * u
            h = 0;
            for (c = 0; c < layer1_size; c++) h += syn0[c + l2] * syn1neg[c + l3]; // h = cos(v, u') = v * u'
        
            if (f - h < margin) {
              obj_w += margin - (f - h);

              // compute context word gradient
              for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
              for (c = 0; c < layer1_size; c++) neu1e[c] += syn0[c + l1] - f * syn1neg[c + l3] + h * syn1neg[c + l3] - syn0[c + l2];
              
              // update positive center word
              for (c = 0; c < layer1_size; c++) grad[c] = syn1neg[c + l3] - f * syn0[c + l1]; // negative Riemannian gradient
              step = 1 - f; // cosine distance, d_cos
              for (c = 0; c < layer1_size; c++) syn0[c + l1] += alpha * step * grad[c];
              g = 0;
              for (c = 0; c < layer1_size; c++) g += syn0[c + l1] * syn0[c + l1];
              g = sqrt(g);
              for (c = 0; c < layer1_size; c++) syn0[c + l1] /= g;

              // update negative center word
              for (c = 0; c < layer1_size; c++) grad[c] = h * syn0[c + l2] - syn1neg[c + l3];
              step = 2 * h; // 2 * negative cosine similarity
              for (c = 0; c < layer1_size; c++) syn0[c + l2] += alpha * step * grad[c];
              g = 0;
              for (c = 0; c < layer1_size; c++) g += syn0[c + l2] * syn0[c + l2];
              g = sqrt(g);
              for (c = 0; c < layer1_size; c++) syn0[c + l2] /= g;

              // update context word
              step = 1 - (f - h);
              for (c = 0; c < layer1_size; c++) syn1neg[c + l3] += alpha * step * neu1e[c];
              g = 0;
              for (c = 0; c < layer1_size; c++) g += syn1neg[c + l3] * syn1neg[c + l3];
              g = sqrt(g);
              for (c = 0; c < layer1_size; c++) syn1neg[c + l3] /= g;
            }
          }
        }
      }

    obj_d = 0;
    l1 = doc * layer1_size; // positive document d
    for (d = 0; d < negative + 1; d++) {
      if (d == 0) {
        l3 = word * layer1_size; // positive center word u
      } else {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        target = word_table[(next_random >> 16) % table_size];
        if (target == 0) target = next_random % (vocab_size - 1) + 1;
        if (target == word) continue;
        l2 = target * layer1_size; // negative center word u'
      
        f = 0;
        for (c = 0; c < layer1_size; c++) f += syn0[c + l3] * syn1doc[c + l1]; // f = cos(u, d) = u * d
        h = 0;
        for (c = 0; c < layer1_size; c++) h += syn0[c + l2] * syn1doc[c + l1]; // h = cos(u', d) = u' * d
    
        if (f - h < margin) {
          obj_d += margin - (f - h);

          // compute document gradient
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
          for (c = 0; c < layer1_size; c++) neu1e[c] += syn0[c + l3] - f * syn1doc[c + l1] + h * syn1doc[c + l1] - syn0[c + l2];

          // update positive center word
          for (c = 0; c < layer1_size; c++) grad[c] = syn1doc[c + l1] - f * syn0[c + l3];
          step = 1 - f;
          for (c = 0; c < layer1_size; c++) syn0[c + l3] += alpha * step * grad[c];
          g = 0;
          for (c = 0; c < layer1_size; c++) g += syn0[c + l3] * syn0[c + l3];
          g = sqrt(g);
          for (c = 0; c < layer1_size; c++) syn0[c + l3] /= g;

          // update negative center word
          for (c = 0; c < layer1_size; c++) grad[c] = h * syn0[c + l2] - syn1doc[c + l1];
          step = 2 * h;
          for (c = 0; c < layer1_size; c++) syn0[c + l2] += alpha * step * grad[c];
          g = 0;
          for (c = 0; c < layer1_size; c++) g += syn0[c + l2] * syn0[c + l2];
          g = sqrt(g);
          for (c = 0; c < layer1_size; c++) syn0[c + l2] /= g;

          // update document
          step = 1 - (f - h);
          for (c = 0; c < layer1_size; c++) syn1doc[c + l1] += alpha * step * neu1e[c];
          g = 0;
          for (c = 0; c < layer1_size; c++) g += syn1doc[c + l1] * syn1doc[c + l1];
          g = sqrt(g);
          for (c = 0; c < layer1_size; c++) syn1doc[c + l1] /= g;
        }
      }
    }

    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  free(grad);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b;
  FILE *fo;
  pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);

  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  
  InitNet();
  InitUnigramTable();
  start = clock();
  
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

  if (word_emb[0] != 0) {
    fo = fopen(word_emb, "wb");
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      for (b = 0; b < layer1_size; b++) {
        fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      }
      fprintf(fo, "\n");
    }
    fclose(fo);
  }
  
  if (context_emb[0] != 0) {
    FILE* fa = fopen(context_emb, "wb");
    fprintf(fa, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fa, "%s ", vocab[a].word);
      for (b = 0; b < layer1_size; b++) {
        fprintf(fa, "%lf ", syn1neg[a * layer1_size + b]);
      }
      fprintf(fa, "\n");
    }
    fclose(fa);
  }

  if (doc_output[0] != 0) {
    FILE* fd = fopen(doc_output, "wb");
    fprintf(fd, "%lld %lld\n", corpus_size, layer1_size);
    for (a = 0; a < corpus_size; a++) {
      fprintf(fd, "%ld ", a);
      for (b = 0; b < layer1_size; b++) {
        fprintf(fd, "%lf ", syn1doc[a * layer1_size + b]);
      }
      fprintf(fd, "\n");
    }
    fclose(fd);
  }
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("Parameters:\n");
    printf("\t-train <file> (mandatory argument)\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-word-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors\n");
    printf("\t-context-output <file>\n");
    printf("\t\tUse <file> to save the resulting word context vectors\n");
    printf("\t-doc-output <file>\n");
    printf("\t\tUse <file> to save the resulting document vectors\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the\n");
    printf("\t\ttraining data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-3)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 2\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads; default is 20\n");
    printf("\t-margin <float>\n");
    printf("\t\tMargin used in loss function to separate positive samples from negative samples; default is 0.15\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations; default is 10\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.04\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-load-emb <file>\n");
    printf("\t\tThe pretrained embeddings will be read from <file>\n");
    printf("\nExamples:\n");
    printf(
        "./jose -train text.txt -word-output jose.txt -size 100 -margin 0.15 -window 5 -sample 1e-3 -negative 2 -iter 10\n\n");
    return 0;
  }
  word_emb[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *) "-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-load-emb", argc, argv)) > 0) strcpy(load_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-word-output", argc, argv)) > 0) strcpy(word_emb, argv[i + 1]);
  if ((i = ArgPos((char *) "-context-output", argc, argv)) > 0) strcpy(context_emb, argv[i + 1]);
  if ((i = ArgPos((char *) "-doc-output", argc, argv)) > 0) strcpy(doc_output, argv[i + 1]);
  if ((i = ArgPos((char *) "-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
  docs = (int *) calloc(corpus_max_size, sizeof(int));
  doc_sizes = (long long *) calloc(corpus_max_size, sizeof(long long));
  if (negative <= 0) {
    printf("ERROR: Nubmer of negative samples must be positive!\n");
    exit(1);
  }
  TrainModel();
  return 0;
}
