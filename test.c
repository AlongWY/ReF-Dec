#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int func000(float numbers[], int size, float threshold) {
    int i, j;

    for (i = 0; i < size; i++)
        for (j = i + 1; j < size; j++)
            if (fabs(numbers[i] - numbers[j]) < threshold)
                return 1;

    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char** func001(const char* paren_string, int* group_count) {
    int length = strlen(paren_string);
    int level = 0;
    int capacity = 10;
    char** groups = malloc(capacity * sizeof(char*));
    char* buffer = malloc(length + 1);
    int buffer_index = 0;
    *group_count = 0;

    for (int i = 0; i < length; ++i) {
        char chr = paren_string[i];
        if (chr == '(') {
            level++;
            buffer[buffer_index++] = chr;
        } else if (chr == ')') {
            level--;
            buffer[buffer_index++] = chr;
            if (level == 0) {
                buffer[buffer_index] = '\0';
                groups[*group_count] = strdup(buffer);
                (*group_count)++;
                if (*group_count >= capacity) {
                    capacity *= 2;
                    groups = realloc(groups, capacity * sizeof(char*));
                }
                buffer_index = 0;
            }
        }
    }

    free(buffer);
    return groups;
}


#include <stdio.h>
#include <math.h>

float func002(float number) {
    return number - (int)number;
}


#include <stdio.h>

int func003(int operations[], int size) {
    int num = 0;
    for (int i = 0; i < size; i++) {
        num += operations[i];
        if (num < 0) return 1;
    }
    return 0;
}


#include <stdio.h>
#include <math.h>

float func004(float numbers[], int size) {
    float sum = 0;
    float avg, msum, mavg;
    int i = 0;

    for (i = 0; i < size; i++)
        sum += numbers[i];

    avg = sum / size;
    msum = 0;

    for (i = 0; i < size; i++)
        msum += fabs(numbers[i] - avg);

    return msum / size;
}


#include <stdio.h>
#include <stdlib.h>

int *func005(const int numbers[], int size, int delimiter, int *out_size) {
    *out_size = size > 0 ? (size * 2) - 1 : 0;
    int *out = (int *)malloc(*out_size * sizeof(int));
    if (size > 0) out[0] = numbers[0];
    for (int i = 1, j = 1; i < size; ++i) {
        out[j++] = delimiter;
        out[j++] = numbers[i];
    }
    return out;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int* func006(const char* paren_string, int* returnSize) {
    int* all_levels = NULL;
    int level = 0, max_level = 0, i = 0, count = 0;
    char chr;
    for (i = 0; paren_string[i] != '\0'; i++) {
        chr = paren_string[i];
        if (chr == '(') {
            level += 1;
            if (level > max_level) max_level = level;
        } else if (chr == ')') {
            level -= 1;
            if (level == 0) {
                all_levels = (int*)realloc(all_levels, sizeof(int) * (count + 1));
                all_levels[count++] = max_level;
                max_level = 0;
            }
        }
    }
    *returnSize = count;
    return all_levels;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char **func007(char **strings, int size, const char *substring, int *out_size) {
    char **out = NULL;
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (strstr(strings[i], substring) != NULL) {
            out = (char **)realloc(out, sizeof(char *) * (count + 1));
            out[count] = strings[i];
            count++;
        }
    }
    *out_size = count;
    return out;
}


#include <stdio.h>

void func008(int *numbers, int size, int *result) {
    int sum = 0, product = 1;
    for (int i = 0; i < size; i++) {
        sum += numbers[i];
        product *= numbers[i];
    }
    result[0] = sum;
    result[1] = product;
}


#include <stdio.h>
#include <stdlib.h>

int *func009(int *numbers, int size) {
    if (size <= 0) {
        return NULL;
    }
    
    int *out = malloc(size * sizeof(int));
    if (!out) {
        return NULL;
    }
    
    int max = numbers[0];
    for (int i = 0; i < size; i++) {
        if (numbers[i] > max) max = numbers[i];
        out[i] = max;
    }
    return out;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *func010(const char *str) {
    int len = strlen(str), i, j;
    char *result = (char *)malloc(2 * len + 1);
    if (!result) {
        return NULL; 
    }

    for (i = 0; i < len; i++) {
        int is_palindrome = 1;
        for (j = 0; j < (len - i) / 2; j++) {
            if (str[i + j] != str[len - 1 - j]) {
                is_palindrome = 0;
                break;
            }
        }
        if (is_palindrome) {
            strncpy(result, str, len);
            for (j = 0; j < i; j++) {
                result[len + j] = str[i - j - 1];
            }
            result[len + i] = '\0';
            return result;
        }
    }

    strncpy(result, str, len);
    for (j = 0; j < len; j++) {
        result[len + j] = str[len - j - 1];
    }
    result[2 * len] = '\0';
    return result;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *func011(const char *a, const char *b) {
    int len_a = strlen(a);
    int len_b = strlen(b);
    int min_len = len_a < len_b ? len_a : len_b;
    char *output = malloc((min_len + 1) * sizeof(char));
    if (!output) return NULL;

    for (int i = 0; i < min_len; i++) {
        output[i] = (a[i] == b[i]) ? '0' : '1';
    }
    output[min_len] = '\0';
    return output;
}


#include <stdio.h>
#include <string.h>

char *func012(char **strings, int count) {
    char *out = "";
    int longest_length = 0;
    for (int i = 0; i < count; i++) {
        int current_length = strlen(strings[i]);
        if (current_length > longest_length) {
            out = strings[i];
            longest_length = current_length;
        }
    }
    return out;
}


#include <stdio.h>

int func013(int a, int b) {
    while (b != 0) {
        int m = a % b;
        a = b;
        b = m;
    }
    return a;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char **func014(const char *str, int *count) {
    int len = strlen(str);
    char **out = malloc(len * sizeof(char *));
    
    char *current = malloc(len + 1);
    current[0] = '\0';

    for (int i = 0; i < len; ++i) {
        size_t current_len = strlen(current);
        current = realloc(current, current_len + 2);
        current[current_len] = str[i];
        current[current_len + 1] = '\0';

        out[i] = malloc(strlen(current) + 1);
        strcpy(out[i], current);
    }
    free(current);
    
    *count = len;
    return out;
}


#include <stdio.h>
#include <stdlib.h>

char *func015(int n) {
    int len = 2; 
    for (int i = 1; i <= n; ++i) {
        len += snprintf(NULL, 0, " %d", i);
    }

    char *out = malloc(len);
    if (!out) {
        return NULL;
    }
    
    char *ptr = out;
    ptr += sprintf(ptr, "0");
    for (int i = 1; i <= n; ++i) {
        ptr += sprintf(ptr, " %d", i);
    }
    return out;
}


#include <stdio.h>
#include <string.h>
#include <ctype.h>

int func016(const char *str) {
    int count = 0;
    int char_map[256] = {0};
    int index;
    
    for (index = 0; str[index]; index++) {
        char ch = tolower((unsigned char)str[index]);
        if (char_map[ch] == 0 && isalpha((unsigned char)ch)) {
            char_map[ch] = 1;
            count++;
        }
    }
    
    return count;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int *func017(const char *music_string, int *count) {
    int *out = NULL;
    int size = 0;
    int capacity = 0;

    char current[3] = "";
    int music_string_length = strlen(music_string) + 1;
    char *temp_music_string = malloc(music_string_length + 1);
    strcpy(temp_music_string, music_string);
    strcat(temp_music_string, " ");

    for (int i = 0; i < music_string_length; i++) {
        if (temp_music_string[i] == ' ') {
            if (strcmp(current, "o") == 0) {
                if (size == capacity) {
                    capacity = capacity > 0 ? 2 * capacity : 4;
                    out = realloc(out, capacity * sizeof(int));
                }
                out[size++] = 4;
            }
            if (strcmp(current, "o|") == 0) {
                if (size == capacity) {
                    capacity = capacity > 0 ? 2 * capacity : 4;
                    out = realloc(out, capacity * sizeof(int));
                }
                out[size++] = 2;
            }
            if (strcmp(current, ".|") == 0) {
                if (size == capacity) {
                    capacity = capacity > 0 ? 2 * capacity : 4;
                    out = realloc(out, capacity * sizeof(int));
                }
                out[size++] = 1;
            }
            strcpy(current, "");
        } else {
            size_t len = strlen(current);
            if (len < sizeof(current) - 1) {
                current[len] = temp_music_string[i];
                current[len + 1] = '\0';
            }
        }
    }
    free(temp_music_string);
    *count = size;
    return out;
}


#include <stdio.h>
#include <string.h>

int func018(const char *str, const char *substring) {
    int out = 0;
    int str_len = strlen(str);
    int sub_len = strlen(substring);
    if (str_len == 0) return 0;
    for (int i = 0; i <= str_len - sub_len; i++) {
        if (strncmp(&str[i], substring, sub_len) == 0)
            out++;
    }
    return out;
}


#include <stdio.h>
#include <string.h>

const char* func019(const char* numbers) {
    int count[10] = {0};
    const char* numto[10] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
    int index, i, j, k;
    static char out[1000]; 
    char current[6]; 

    index = 0;
    if (*numbers) {
        do {
            for (i = 0; numbers[i] != ' ' && numbers[i] != '\0'; ++i) {
                current[i] = numbers[i];
            }
            current[i] = '\0';
            for (j = 0; j < 10; ++j) {
                if (strcmp(current, numto[j]) == 0) {
                    count[j]++;
                    break;
                }
            }
            numbers += i + 1;
        } while (numbers[-1]);
    }

    for (i = 0; i < 10; ++i) {
        for (j = 0; j < count[i]; ++j) {
            for (k = 0; numto[i][k] != '\0'; ++k, ++index) {
                out[index] = numto[i][k];
            }
            out[index++] = ' '; 
        }
    }

    if (index > 0) {
        out[index - 1] = '\0'; 
    } else {
        out[0] = '\0';
    }

    return out;
}


#include <stdio.h>
#include <math.h>
#include <float.h>

void func020(float numbers[], int size, float out[2]) {
    float min_diff = FLT_MAX;
    int i, j;

    out[0] = numbers[0];
    out[1] = numbers[1];

    for (i = 0; i < size; i++) {
        for (j = i + 1; j < size; j++) {
            float diff = fabs(numbers[i] - numbers[j]);
            if (diff < min_diff) {
                min_diff = diff;
                out[0] = numbers[i];
                out[1] = numbers[j];
            }
        }
    }

    if (out[0] > out[1]) {
        float temp = out[0];
        out[0] = out[1];
        out[1] = temp;
    }
}


#include <stdio.h>
#include <math.h>

void func021(float *numbers, int size) {
    float min = numbers[0], max = numbers[0];
    for (int i = 1; i < size; i++) {
        if (numbers[i] < min) min = numbers[i];
        if (numbers[i] > max) max = numbers[i];
    }
    for (int i = 0; i < size; i++) {
        numbers[i] = (numbers[i] - min) / (max - min);
    }
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int *func022(const char *values, int *size) {
    static int out[256];
    int count = 0;
    const char *start = values;
    char *end;
    while (*start) {
        while (*start && !isdigit(*start) && *start != '-') {
            start++;
        }
        if (!*start) {
            break;
        }
        int val = (int) strtol(start, &end, 10);
        if (start != end && (*end == ',' || *end == '\0')) {
            out[count++] = val;
        } else {
            while (*end && *end != ',') {
                end++;
            }
        }
        start = end;
    }
    *size = count;
    return out;
}


#include <stdio.h>

int func023(const char *str) {
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    return length;
}


#include <stdio.h>

int func024(int n) {
    for (int i = 2; i * i <= n; i++)
        if (n % i == 0) return n / i;
    return 1;
}


#include <stdio.h>
#include <stdlib.h>

int* func025(int n, int* size) {
    int* out = malloc(sizeof(int) * 64);
    *size = 0;
    for (int i = 2; i * i <= n; i++) {
        while (n % i == 0) {
            n = n / i;
            out[(*size)++] = i;
        }
    }
    if (n > 1) {
        out[(*size)++] = n;
    }
    return out;
}


#include <stdio.h>
#include <stdlib.h>

int* func026(int* numbers, int size, int* new_size) {
    int* out = (int*)malloc(size * sizeof(int));
    int* has1 = (int*)calloc(size, sizeof(int));
    int* has2 = (int*)calloc(size, sizeof(int));
    int has1_count = 0;
    int has2_count = 0;
    int out_count = 0;

    for (int i = 0; i < size; i++) {
        int num = numbers[i];
        int in_has2 = 0;

        for (int j = 0; j < has2_count; j++) {
            if (has2[j] == num) {
                in_has2 = 1;
                break;
            }
        }
        if (in_has2) continue;

        int in_has1 = 0;
        for (int j = 0; j < has1_count; j++) {
            if (has1[j] == num) {
                in_has1 = 1;
                break;
            }
        }
        if (in_has1) {
            has2[has2_count++] = num;
        } else {
            has1[has1_count++] = num;
        }
    }

    for (int i = 0; i < size; i++) {
        int num = numbers[i];
        int in_has2 = 0;
        for (int j = 0; j < has2_count; j++) {
            if (has2[j] == num) {
                in_has2 = 1;
                break;
            }
        }
        if (!in_has2) {
            out[out_count++] = num;
        }
    }

    *new_size = out_count;
    free(has1);
    free(has2);
    return out;
}


#include <stdio.h>
#include <string.h>

void func027(const char* str, char* out) {
    int length = strlen(str);
    for (int i = 0; i < length; i++) {
        char w = str[i];
        if (w >= 'a' && w <= 'z') {
            w -= 32;
        } else if (w >= 'A' && w <= 'Z') {
            w += 32;
        }
        out[i] = w;
    }
    out[length] = '\0';
}


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char* func028(char** strings, int count) {
    int length = 0;
    for (int i = 0; i < count; i++) {
        length += strlen(strings[i]);
    }
    
    char* out = (char*)malloc(length + 1);
    if (!out) {
        return NULL; 
    }
    
    out[0] = '\0';

    for (int i = 0; i < count; i++) {
        strcat(out, strings[i]);
    }
    
    return out;
}


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int func029(char** strings, int count, const char* prefix, char*** out) {
    int prefix_length = strlen(prefix);
    *out = (char**)malloc(count * sizeof(char*));
    int out_count = 0;

    for (int i = 0; i < count; i++) {
        if (strncmp(strings[i], prefix, prefix_length) == 0) {
            (*out)[out_count++] = strings[i];
        }
    }

    return out_count;
}


#include <stdio.h>
#include <stdlib.h>

float* func030(const float* l, int count, int* out_count) {
    float* out = (float*)malloc(count * sizeof(float));
    *out_count = 0;

    for (int i = 0; i < count; i++) {
        if (l[i] > 0) {
            out[(*out_count)++] = l[i];
        }
    }

    return out;
}


#include <stdbool.h>

bool func031(long long n) {
    if (n < 2) return false;
    for (long long i = 2; i * i <= n; i++)
        if (n % i == 0) return false;
    return true;
}


#include <stdio.h>
#include <math.h>

double func032(const double *xs, int size) {
    double ans = 0.0;
    double value, driv, x_pow;
    int i;

    value = xs[0];
    for (i = 1; i < size; i++) {
        x_pow = 1.0;
        for (int j = 0; j < i; j++) {
            x_pow *= ans;
        }
        value += xs[i] * x_pow;
    }

    while (fabs(value) > 1e-6) {
        driv = 0.0;
        for (i = 1; i < size; i++) {
            x_pow = 1.0;
            for (int j = 1; j < i; j++) {
                x_pow *= ans;
            }
            driv += i * xs[i] * x_pow;
        }
        ans = ans - value / driv;

        value = xs[0];
        for (i = 1; i < size; i++) {
            x_pow = 1.0;
            for (int j = 0; j < i; j++) {
                x_pow *= ans;
            }
            value += xs[i] * x_pow;
        }
    }

    return ans;
}


#include <stdio.h>
#include <stdlib.h>

void func033(int *l, int size, int *out) {
    int *third = malloc((size / 3 + 1) * sizeof(int));
    int i, k = 0, third_size = 0;

    for (i = 0; i * 3 < size; i++) {
        third[i] = l[i * 3];
        third_size++;
    }

    for (i = 0; i < third_size - 1; i++) {
        int min_idx = i;
        for (k = i + 1; k < third_size; k++) {
            if (third[k] < third[min_idx])
                min_idx = k;
        }
        if (min_idx != i) {
            int temp = third[i];
            third[i] = third[min_idx];
            third[min_idx] = temp;
        }
    }

    for (i = 0; i < size; i++) {
        if (i % 3 == 0) {
            out[i] = third[i / 3];
        } else {
            out[i] = l[i];
        }
    }

    free(third);
}


#include <stdio.h>
#include <stdlib.h>

int *func034(int *l, int size, int *out_size) {
    int *out = malloc(size * sizeof(int));
    int found, out_count = 0, i, j;
    for (i = 0; i < size; i++) {
        found = 0;
        for (j = 0; j < out_count; j++) {
            if (l[i] == out[j]) {
                found = 1;
                break;
            }
        }
        if (!found) {
            out[out_count++] = l[i];
        }
    }

    for (i = 0; i < out_count - 1; i++) {
        for (j = i + 1; j < out_count; j++) {
            if (out[i] > out[j]) {
                int temp = out[i];
                out[i] = out[j];
                out[j] = temp;
            }
        }
    }

    *out_size = out_count;
    return out;
}


#include <stdio.h>

float func035(float *l, int size) {
    float max = -10000;
    for (int i = 0; i < size; i++)
        if (max < l[i]) max = l[i];
    return max;
}


#include <stdio.h>

int func036(int n) {
    int count = 0;
    for (int i = 0; i < n; i++)
        if (i % 11 == 0 || i % 13 == 0) {
            int q = i;
            while (q > 0) {
                if (q % 10 == 7) count += 1;
                q = q / 10;
            }
        }
    return count;
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void func037(float *l, int size, float *out) {
    float *even = malloc((size / 2 + 1) * sizeof(float));
    int i, j, even_count = 0;

    for (i = 0; i < size; i += 2) {
        even[even_count++] = l[i];
    }

    for (i = 0; i < even_count - 1; i++) {
        for (j = 0; j < even_count - i - 1; j++) {
            if (even[j] > even[j + 1]) {
                float temp = even[j];
                even[j] = even[j + 1];
                even[j + 1] = temp;
            }
        }
    }

    // Merging even-indexed sorted and odd-indexed as they are
    for (i = 0; i < size; i++) {
        if (i % 2 == 0) {
            out[i] = even[i / 2];
        } else {
            out[i] = l[i];
        }
    }

    free(even);
}


#include <stdio.h>
#include <string.h>

void func038(char *s, int encode) {
    int l = strlen(s);
    int num = (l + 2) / 3;
    char x[4];

    for (int i = 0; i < num; ++i) {
        int len = (i * 3 + 3 <= l) ? 3 : l - i * 3;
        strncpy(x, s + i * 3, len);
        x[len] = '\0';

        if (len == 3) {
            if (encode) {
                char temp = x[2];
                x[2] = x[1];
                x[1] = x[0];
                x[0] = temp;
            } else {
                char temp = x[0];
                x[0] = x[1];
                x[1] = x[2];
                x[2] = temp;
            }
        }
        strncpy(s + i * 3, x, len);
    }
}


#include <stdio.h>

int func039(int n) {
    int f1 = 1, f2 = 2, m;
    int count = 0;
    while (count < n) {
        f1 = f1 + f2;
        m = f1; f1 = f2; f2 = m;
        int isprime = 1;
        for (int w = 2; w * w <= f1; w++) {
            if (f1 % w == 0) {
                isprime = 0; break;
            }
        }
        if (isprime) count += 1;
        if (count == n) return f1;
    }
    return 0;
}


#include <stdio.h>
#include <stdbool.h>

bool func040(int *l, int size) {
    for (int i = 0; i < size; i++)
        for (int j = i + 1; j < size; j++)
            for (int k = j + 1; k < size; k++)
                if (l[i] + l[j] + l[k] == 0) return true;
    return false;
}


#include <stdio.h>

int func041(int n) {
    return n * n;
}


#include <stdio.h>

void func042(int *l, int size) {
    for (int i = 0; i < size; i++)
        l[i] += 1;
}


#include <stdio.h>
#include <stdbool.h>

bool func043(int *l, int size) {
    for (int i = 0; i < size; i++)
        for (int j = i + 1; j < size; j++)
            if (l[i] + l[j] == 0) return true;
    return false;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void func044(int x, int base, char *out) {
    int index = 0;
    char temp[33];
    while (x > 0) {
        temp[index++] = (x % base) + '0';
        x = x / base;
    }
    int j = 0;
    while(index > 0) {
        out[j++] = temp[--index];
    }
    out[j] = '\0';
}


#include <stdio.h>
#include <math.h>

float func045(float a, float h) {
    return (a * h) * 0.5;
}


#include <stdio.h>

int func046(int n) {
    int f[100];
    f[0] = 0;
    f[1] = 0;
    f[2] = 2;
    f[3] = 0;
    for (int i = 4; i <= n; i++) {
        f[i] = f[i - 1] + f[i - 2] + f[i - 3] + f[i - 4];
    }
    return f[n];
}


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

float func047(float *l, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            if (l[i] > l[j]) {
                float temp = l[i];
                l[i] = l[j];
                l[j] = temp;
            }
        }
    }
    if (size % 2 == 1) return l[size / 2];
    return 0.5 * (l[size / 2] + l[size / 2 - 1]);
}


#include <stdio.h>
#include <string.h>
#include <stdbool.h>

bool func048(const char *text) {
    int len = strlen(text);
    for (int i = 0; i < len / 2; i++) {
        if (text[i] != text[len - 1 - i]) {
            return false;
        }
    }
    return true;
}


#include <stdio.h>

int func049(int n, int p) {
    int out = 1;
    for (int i = 0; i < n; i++)
        out = (out * 2) % p;
    return out;
}


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void func050(char *s, int encode) {
    int shift = encode ? 5 : 21;
    size_t len = strlen(s);
    for (size_t i = 0; i < len; i++) {
        int w = ((s[i] - 'a' + shift) % 26) + 'a';
        s[i] = (char)w;
    }
}


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void func051(char *text) {
    const char *vowels = "AEIOUaeiou";
    char *out = text;
    while (*text != '\0') {
        if (strchr(vowels, *text) == NULL) {
            *out++ = *text;
        }
        text++;
    }
    *out = '\0';
}


#include <stdio.h>
#include <stdbool.h>

bool func052(int *l, int size, int t) {
    for (int i = 0; i < size; i++)
        if (l[i] >= t) return false;
    return true;
}


#include <stdio.h>

int func053(int x, int y) {
    return x + y;
}


#include <stdio.h>
#include <string.h>
#include <stdbool.h>

bool func054(const char *s0, const char *s1) {
    int len0 = strlen(s0), len1 = strlen(s1);
    for (int i = 0; i < len0; i++) {
        bool found = false;
        for (int j = 0; j < len1; j++) {
            if (s0[i] == s1[j]) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    for (int i = 0; i < len1; i++) {
        bool found = false;
        for (int j = 0; j < len0; j++) {
            if (s1[i] == s0[j]) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}


#include <stdio.h>

int func055(int n) {
    int f[1000] = {0};
    f[0] = 0;
    f[1] = 1;
    for (int i = 2; i <= n; i++)
        f[i] = f[i - 1] + f[i - 2];
    return f[n];
}


#include <stdio.h>
#include <stdbool.h>
#include <string.h>

bool func056(const char *brackets) {
    int level = 0;
    int i = 0;
    while (brackets[i]) {
        if (brackets[i] == '<') level++;
        if (brackets[i] == '>') level--;
        if (level < 0) return false;
        i++;
    }
    if (level != 0) return false;
    return true;
}


#include <stdio.h>
#include <stdbool.h>

bool func057(float *l, int size) {
    int incr, decr;
    incr = decr = 0;
    
    for (int i = 1; i < size; i++) {
        if (l[i] > l[i - 1]) incr = 1;
        if (l[i] < l[i - 1]) decr = 1;
    }
    if (incr + decr == 2) return false;
    return true;
}


#include <stdio.h>
#include <stdlib.h>

int *func058(int *l1, int size1, int *l2, int size2, int *out_size) {
    int *out = malloc(size1 * sizeof(int));
    int k = 0, i, j, m;

    for (i = 0; i < size1; i++) {
        int exists_in_out = 0;
        for (m = 0; m < k; m++) {
            if (out[m] == l1[i]) {
                exists_in_out = 1;
                break;
            }
        }
        if (!exists_in_out) {
            for (j = 0; j < size2; j++) {
                if (l1[i] == l2[j]) {
                    out[k++] = l1[i];
                    break;
                }
            }
        }
    }

    for (i = 0; i < k - 1; i++) {
        for (j = 0; j < k - i - 1; j++) {
            if (out[j] > out[j + 1]) {
                int temp = out[j];
                out[j] = out[j + 1];
                out[j + 1] = temp;
            }
        }
    }

    *out_size = k;
    return out;
}


#include <stdio.h>

int func059(int n) {
    for (int i = 2; i * i <= n; i++)
        while (n % i == 0 && n > i) n = n / i;
    return n;
}


#include <stdio.h>

int func060(int n) {
    return n * (n + 1) / 2;
}


#include <stdio.h>
#include <stdbool.h>
#include <string.h>

bool func061(const char *brackets) {
    int level = 0;
    for (int i = 0; i < strlen(brackets); i++) {
        if (brackets[i] == '(') level += 1;
        if (brackets[i] == ')') level -= 1;
        if (level < 0) return false;
    }
    return level == 0;
}


#include <stdio.h>

void func062(const float *xs, int xs_size, float *out) {
    for (int i = 1; i < xs_size; i++) {
        out[i - 1] = i * xs[i];
    }
}


#include <stdio.h>

int func063(int n) {
    int ff[100] = {0};
    ff[1] = 0;
    ff[2] = 1;
    for (int i = 3; i <= n; ++i) {
        ff[i] = ff[i - 1] + ff[i - 2] + ff[i - 3];
    }
    return ff[n];
}


#include <stdio.h>
#include <string.h>
#include <ctype.h>

int func064(const char *s) {
    const char *vowels = "aeiouAEIOU";
    int count = 0;
    int length = strlen(s);
    
    for (int i = 0; i < length; i++) {
        if (strchr(vowels, s[i])) {
            count++;
        }
    }
    
    if (length > 0 && (s[length - 1] == 'y' || s[length - 1] == 'Y')) {
        count++;
    }
    
    return count;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* func065(int x, int shift) {
    static char xs[50];
    sprintf(xs, "%d", x);
    int len = strlen(xs);

    if (len < shift) {
        for (int i = 0; i < len / 2; i++) {
            char temp = xs[i];
            xs[i] = xs[len - 1 - i];
            xs[len - 1 - i] = temp;
        }
    } else {
        char temp[50];
        strcpy(temp, xs + len - shift);
        temp[shift] = '\0';
        strncat(temp, xs, len - shift);
        strcpy(xs, temp);
    }

    return xs;
}


#include <stdio.h>
#include <string.h>

int func066(const char *s) {
    int sum = 0;
    for (int i = 0; s[i] != '\0'; i++)
        if (s[i] >= 'A' && s[i] <= 'Z')
            sum += s[i];
    return sum;
}


#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

int func067(const char *s, int n) {
    char num1[10] = "";
    char num2[10] = "";
    int is12 = 0, j = 0;

    for (int i = 0; s[i] != '\0'; i++) {
        if (isdigit(s[i])) {
            if (is12 == 0) {
                num1[j++] = s[i];
            } else {
                num2[j++] = s[i];
            }
        } else {
            if (is12 == 0 && j > 0) {
                is12 = 1;
                j = 0;
            }
        }
    }
    return n - atoi(num1) - atoi(num2);
}


#include <stdio.h>
#include <limits.h>

int *func068(int arr[], int length, int output[2]) {
    int smallestEven = INT_MAX;
    int index = -1;
    
    for (int i = 0; i < length; ++i) {
        if (arr[i] % 2 == 0 && (arr[i] < smallestEven || index == -1)) {
            smallestEven = arr[i];
            index = i;
        }
    }
    
    if (index == -1) {
        return NULL;
    }

    output[0] = smallestEven;
    output[1] = index;
    return output;
}


#include <stdio.h>
#include <stdlib.h>

int func069(int *lst, int size) {
    int *freq = (int *)calloc(size + 1, sizeof(int));
    int max = -1;

    for (int i = 0; i < size; i++) {
        freq[lst[i]]++;
        if ((freq[lst[i]] >= lst[i]) && (lst[i] > max)) {
            max = lst[i];
        }
    }

    free(freq);
    return max;
}


#include <stdio.h>
#include <stdlib.h>

void func070(int *lst, int size, int *out) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            if (lst[i] > lst[j]) {
                int temp = lst[i];
                lst[i] = lst[j];
                lst[j] = temp;
            }
        }
    }

    int l = 0, r = size - 1;
    int index = 0;

    while (l <= r) {
        if (l == r) {
            out[index++] = lst[l++];
        } else {
            out[index++] = lst[l++];
            out[index++] = lst[r--];
        }
    }
}


#include <stdio.h>
#include <math.h>

float func071(float a, float b, float c) {
if (a + b <= c || a + c <= b || b + c <= a) return -1;
float s = (a + b + c) / 2;
float area = sqrtf(s * (s - a) * (s - b) * (s - c));
return roundf(area * 100) / 100;
}


#include <stdio.h>
#include <stdbool.h>

bool func072(int q[], int size, int w) {
    int sum = 0;
    for (int i = 0; i < size / 2; i++) {
        if (q[i] != q[size - 1 - i]) return false;
        sum += q[i] + q[size - 1 - i];
    }
    if (size % 2 == 1) sum += q[size / 2];
    return sum <= w;
}


#include <stdio.h>

int func073(int arr[], int size) {
    int out = 0;
    for (int i = 0; i < size / 2; i++) {
        if (arr[i] != arr[size - 1 - i]) {
            out++;
        }
    }
    return out;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char** func074(char** arr1, int n1, char** arr2, int n2){

  int i, sum1 = 0, sum2 = 0;

  for(i=0; i<n1; i++){
    sum1 += strlen(arr1[i]);
  }
  
  for(i=0; i<n2; i++){
    sum2 += strlen(arr2[i]); 
  }

  if(sum1 < sum2){
    return arr1;
  }
  else if(sum1 > sum2){
    return arr2;
  }
  else{
    return arr1;
  }

}


#include <stdio.h>
#include <stdlib.h>

int func075(int a) {
    if (a < 2) return 0;
    int num = 0;
    for (int i = 2; i * i <= a; i++) {
        while (a % i == 0) {
            a = a / i;
            num++;
        }
    }
    if (a > 1) num++;
    return num == 3;
}


#include <stdio.h>

int func076(int x, int n) {
    int p = 1, count = 0;
    while (p <= x && count < 100) {
        if (p == x) return 1;
        p = p * n; count += 1;
    }
    return 0;
}


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int func077(int a) {
    for (int i = 0; i * i * i <= abs(a); i++)
        if (i * i * i == abs(a)) return 1;
    return 0;
}


#include <stdio.h>
#include <string.h>

int func078(const char* num) {
    const char* key = "2357BD";
    int out = 0;
    for (int i = 0; i < strlen(num); i++) {
        if (strchr(key, num[i])) out += 1;
    }
    return out;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* func079(int decimal) {
    char* out = malloc(64);
    if (!out) {
        return NULL;
    }
    int index = 62;
    out[63] = '\0';

    if (decimal == 0) {
        out[index--] = '0';
    } else {
        while (decimal > 0) {
            out[index--] = '0' + (decimal % 2);
            decimal /= 2;
        }
    }

    out[index--] = 'b';
    out[index--] = 'd';

    int start = index + 1;
    int len = 62 - start;
    char* formatted_out = malloc(len + 3);
    if (!formatted_out) {
        free(out);
        return NULL;
    }

    strcpy(formatted_out, &out[start]);

    strcat(formatted_out, "db");

    free(out);

    return formatted_out;
}


#include <string.h>
#include <stdbool.h>

bool func080(const char* s) {
    if (strlen(s) < 3) return false;
    for (int i = 2; i < strlen(s); i++)
        if (s[i] == s[i-1] || s[i] == s[i-2]) return false;
    return true;
}


#include <stdio.h>
#include <stdlib.h>

char** func081(float* grades, int size) {
    char** out = malloc(size * sizeof(char*));
    for (int i = 0; i < size; ++i) {
        if (grades[i] >= 3.9999) out[i] = "A+";
        else if (grades[i] > 3.7001) out[i] = "A";
        else if (grades[i] > 3.3001) out[i] = "A-";
        else if (grades[i] > 3.0001) out[i] = "B+";
        else if (grades[i] > 2.7001) out[i] = "B";
        else if (grades[i] > 2.3001) out[i] = "B-";
        else if (grades[i] > 2.0001) out[i] = "C+";
        else if (grades[i] > 1.7001) out[i] = "C";
        else if (grades[i] > 1.3001) out[i] = "C-";
        else if (grades[i] > 1.0001) out[i] = "D+";
        else if (grades[i] > 0.7001) out[i] = "D";
        else if (grades[i] > 0.0001) out[i] = "D-";
        else out[i] = "E";
    }
    return out;
}


#include <stdio.h>
#include <string.h>
#include <stdbool.h>

bool func082(const char* str) {
    int l = strlen(str);
    if (l < 2) return false;
    for (int i = 2; i * i <= l; i++) {
        if (l % i == 0) return false;
    }
    return true;
}


#include <stdio.h>

int func083(int n) {
    if (n < 1) return 0;
    if (n == 1) return 1;
    int out = 18;
    for (int i = 2; i < n; i++)
        out = out * 10;
    return out;
}


#include <stdio.h>
#include <stdlib.h>

char* func084(int N) {
    char str[6];
    sprintf(str, "%d", N);
    int sum = 0;
    for (int i = 0; str[i] != '\0'; i++)
        sum += str[i] - '0';

    char* bi = malloc(33);
    int index = 0;
    if (sum == 0) {
        bi[index++] = '0';
    } else {
        while (sum > 0) {
            bi[index++] = (sum % 2) + '0';
            sum /= 2;
        }
    }
    bi[index] = '\0';

    for (int i = 0; i < index / 2; i++) {
        char temp = bi[i];
        bi[i] = bi[index - i - 1];
        bi[index - i - 1] = temp;
    }

    return bi;
}


#include <stdio.h>

int func085(int lst[], int size) {
    int sum = 0;
    for (int i = 0; i * 2 + 1 < size; i++)
        if (lst[i * 2 + 1] % 2 == 0) sum += lst[i * 2 + 1];
    return sum;
}


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char* func086(const char* s) {
    int len = strlen(s);
    char* out = malloc(len + 2); 
    char current[51];
    int out_index = 0, current_index = 0;

    for (int i = 0; i <= len; i++) {
        if (s[i] == ' ' || s[i] == '\0') {
            for (int j = 0; j < current_index - 1; j++) {
                for (int k = j + 1; k < current_index; k++) {
                    if (current[j] > current[k]) {
                        char temp = current[j];
                        current[j] = current[k];
                        current[k] = temp;
                    }
                }
            }
            if (out_index > 0) out[out_index++] = ' ';
            for (int j = 0; j < current_index; j++) {
                out[out_index++] = current[j];
            }
            current_index = 0;
        } else {
            current[current_index++] = s[i];
        }
    }

    out[out_index] = '\0';
    return out;
}


#include <stdio.h>
#include <stdlib.h>

int **func087(int **lst, int lst_size, int *row_sizes, int x, int *return_size) {
    int **out = (int **)malloc(100 * sizeof(int *));
    int count = 0;
    for (int i = 0; i < lst_size; i++) {
        for (int j = row_sizes[i] - 1; j >= 0; j--) {
            if (lst[i][j] == x) {
                out[count] = (int *)malloc(2 * sizeof(int));
                out[count][0] = i;
                out[count][1] = j;
                count++;
            }
        }
    }
    *return_size = count;
    return out;
}


#include <stdio.h>
#include <stdlib.h>

void func088(int *array, int size, int **out_array, int *out_size) {
    *out_size = size;
    if (size == 0) {
        *out_array = NULL;
        return;
    }

    *out_array = (int *)malloc(sizeof(int) * size);
    if (*out_array == NULL) {
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        (*out_array)[i] = array[i];
    }

    int shouldSortAscending = (array[0] + array[size - 1]) % 2 == 1;

    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            if (shouldSortAscending) {
                if ((*out_array)[i] > (*out_array)[j]) {
                    int temp = (*out_array)[i];
                    (*out_array)[i] = (*out_array)[j];
                    (*out_array)[j] = temp;
                }
            } else {
                if ((*out_array)[i] < (*out_array)[j]) {
                    int temp = (*out_array)[i];
                    (*out_array)[i] = (*out_array)[j];
                    (*out_array)[j] = temp;
                }
            }
        }
    }
}


#include <stdio.h>
#include <string.h>

void func089(const char *s, char *out) {
    int i;
    for (i = 0; s[i] != '\0'; i++) {
        int w = ((int)s[i] - 'a' + 4) % 26 + 'a';
        out[i] = (char)w;
    }
    out[i] = '\0';
}


#include <stdio.h>
#include <limits.h>

int func090(int *lst, int size) {
    if (size < 2) return -1;

    int first = INT_MAX, second = INT_MAX;
    for (int i = 0; i < size; ++i) {
        if (lst[i] < first) {
            second = first;
            first = lst[i];
        } else if (lst[i] < second && lst[i] != first) {
            second = lst[i];
        }
    }

    if (second == INT_MAX) return -1;
    return second;
}


#include <stdio.h>
#include <string.h>
#include <ctype.h>

int func091(const char *S) {
    int isstart = 1;
    int isi = 0;
    int sum = 0;
    for (int i = 0; S[i] != '\0'; i++) {
        if (isspace(S[i]) && isi) {
            isi = 0;
            sum += 1;
        }
        if (S[i] == 'I' && isstart) {
            isi = 1;
        } else if (!isspace(S[i])) {
            isi = 0;
        }
        if (!isspace(S[i])) {
            isstart = 0;
        }
        if (S[i] == '.' || S[i] == '?' || S[i] == '!') {
            isstart = 1;
        }
    }
    return sum;
}


#include <stdio.h>
#include <math.h>

int func092(float a, float b, float c) {
    if (roundf(a) != a) return 0;
    if (roundf(b) != b) return 0;
    if (roundf(c) != c) return 0;
    if ((a + b == c) || (a + c == b) || (b + c == a)) return 1;
    return 0;
}


#include <stdio.h>
#include <ctype.h>
#include <string.h>

void func093(const char* message, char* out) {
    const char* vowels = "aeiouAEIOU";
    int i, j;
    
    for (i = 0; message[i] != '\0'; ++i) {
        char w = message[i];
        if (islower(w)) {
            w = toupper(w);
        } else if (isupper(w)) {
            w = tolower(w);
        }
        
        for (j = 0; vowels[j] != '\0'; ++j) {
            if (w == vowels[j]) {
                if (j < 10) {
                    w = w + 2;
                }
                break;
            }
        }
        out[i] = w;
    }
    out[i] = '\0';
}


#include <stdio.h>

int func094(int lst[], int size) {
    int largest = 0, sum = 0, num, temp;

    for (int i = 0; i < size; ++i) {
        num = lst[i];
        if (num > 1) {
            int prime = 1;
            for (int j = 2; j * j <= num; ++j) {
                if (num % j == 0) {
                    prime = 0;
                    break;
                }
            }
            if (prime) {
                largest = num > largest ? num : largest;
            }
        }
    }

    while (largest > 0) {
        sum += largest % 10;
        largest /= 10;
    }

    return sum;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int func095(char* dict[][2], int size) {
    if (size == 0) return 0;
    int has_lower = 0, has_upper = 0;
    for (int i = 0; i < size; ++i) {
        char* key = dict[i][0];
        for (int j = 0; key[j]; ++j) {
            if (!isalpha((unsigned char)key[j])) return 0;
            if (isupper((unsigned char)key[j])) has_upper = 1;
            if (islower((unsigned char)key[j])) has_lower = 1;
            if (has_upper + has_lower == 2) return 0;
        }
    }
    return 1;
}


#include <stdio.h>
#include <stdlib.h>

int *func096(int n, int *count) {
    int *out = malloc(n * sizeof(int));
    *count = 0;
    int i, j, isp, k;

    for (i = 2; i < n; i++) {
        isp = 1;
        for (j = 0; j < *count; j++) {
            k = out[j];
            if (k * k > i) break;
            if (i % k == 0) {
                isp = 0;
                break;
            }
        }
        if (isp) {
            out[*count] = i;
            (*count)++;
        }
    }
    return out;
}


#include <stdio.h>
#include <stdlib.h>

int func097(int a, int b) {
    return (abs(a) % 10) * (abs(b) % 10);
}


#include <stdio.h>
#include <string.h>

int func098(const char *s) {
    const char *uvowel = "AEIOU";
    int count = 0;
    for (int i = 0; s[i] != '\0' && i * 2 < strlen(s); i++) {
        if (strchr(uvowel, s[i * 2]) != NULL) {
            count += 1;
        }
    }
    return count;
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int func099(const char *value) {
    double w;
    w = atof(value);
    return (int)(w < 0 ? ceil(w - 0.5) : floor(w + 0.5));
}


#include <stdio.h>
#include <stdlib.h>

int* func100(int n) {
    int* out = (int*)malloc(n * sizeof(int));
    *out = n;
    for (int i = 1; i < n; i++)
        *(out + i) = *(out + i - 1) + 2;
    return out;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char** func101(const char* s, int* count) {
    int capacity = 10;
    char** out = malloc(capacity * sizeof(char*));
    char* current = malloc(strlen(s) + 1);
    int word_count = 0;
    int current_length = 0;

    for (int i = 0; s[i]; i++) {
        if (s[i] == ' ' || s[i] == ',') {
            if (current_length > 0) {
                current[current_length] = '\0';
                out[word_count++] = strdup(current);
                current_length = 0;

                if (word_count >= capacity) {
                    capacity *= 2;
                    out = realloc(out, capacity * sizeof(char*));
                }
            }
        } else {
            current[current_length++] = s[i];
        }
    }

    if (current_length > 0) {
        current[current_length] = '\0';
        out[word_count++] = strdup(current);
    }

    free(current);
    *count = word_count;
    return out;
}


#include <stdio.h>

int func102(int x, int y) {
    if (y < x) return -1;
    if (y == x && y % 2 == 1) return -1;
    if (y % 2 == 1) return y - 1;
    return y;
}


#include <stdio.h>
#include <stdlib.h>

char* func103(int n, int m) {
    if (n > m) return "-1";
    int num = (m + n) / 2;
    char* out = (char*)malloc(33);
    out[0] = '\0';

    int index = 32;
    out[index--] = '\0';

    do {
        out[index--] = '0' + num % 2;
        num /= 2;
    } while (num > 0);

    return &out[index + 1]; 
}


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

void func104(int *x, int size, int **out, int *out_size) {
    *out = malloc(size * sizeof(int));
    *out_size = 0;

    for (int i = 0; i < size; i++) {
        int num = x[i];
        bool has_even_digit = false;
        if (num == 0) has_even_digit = true;

        while (num > 0 && !has_even_digit) {
            if (num % 2 == 0) has_even_digit = true;
            num = num / 10;
        }

        if (!has_even_digit) {
            (*out)[*out_size] = x[i];
            (*out_size)++;
        }
    }

    for (int i = 0; i < *out_size - 1; i++) {
        for (int j = 0; j < *out_size - i - 1; j++) {
            if ((*out)[j] > (*out)[j + 1]) {
                int temp = (*out)[j];
                (*out)[j] = (*out)[j + 1];
                (*out)[j + 1] = temp;
            }
        }
    }
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void func105(int *arr, int size, char ***out, int *out_size) {
    char *names[] = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
    int i, j;

    for (i = 0; i < size - 1; i++) {
        for (j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    *out_size = 0;
    for (i = size - 1; i >= 0; i--) {
        if (arr[i] >= 1 && arr[i] <= 9) {
            (*out_size)++;
        }
    }

    *out = malloc(*out_size * sizeof(char *));

    for (i = size - 1, j = 0; i >= 0 && j < *out_size; i--) {
        if (arr[i] >= 1 && arr[i] <= 9) {
            (*out)[j++] = names[arr[i]];
        }
    }
}


#include <stdio.h>
#include <stdlib.h>

int* func106(int n) {
    int* out = (int*)malloc(n * sizeof(int));
    int sum = 0, prod = 1;
    for (int i = 1; i <= n; i++) {
        sum += i;
        prod *= i;
        if (i % 2 == 0) out[i - 1] = prod;
        else out[i - 1] = sum;
    }
    return out;
}


#include <stdio.h>
#include <stdlib.h>

int* func107(int n) {
    int* counts = (int*)malloc(2 * sizeof(int));
    counts[0] = 0;
    counts[1] = 0;

    for (int i = 1; i <= n; i++) {
        int reversed = 0, original = i;

        int number = i;
        while (number != 0) {
            reversed = reversed * 10 + number % 10;
            number /= 10;
        }

        if (original == reversed) {
            if (original % 2 == 0) counts[0]++;
            else counts[1]++;
        }
    }

    return counts;
}


#include <stdio.h>
#include <stdlib.h>

int func108(int *n, int size) {
    int num = 0;
    for (int i = 0; i < size; i++) {
        if (n[i] > 0) {
            num += 1;
        } else {
            int sum = 0;
            int w = abs(n[i]);
            while (w >= 10) {
                sum += w % 10;
                w = w / 10;
            }
            sum -= w;
            if (sum > 0) num += 1;
        }
    }
    return num;
}


#include <stdio.h>
#include <stdbool.h>

bool func109(int *arr, int size) {
    int num = 0;
    if (size == 0) return true;
    for (int i = 1; i < size; i++)
        if (arr[i] < arr[i - 1]) num += 1;
    if (arr[size - 1] > arr[0]) num += 1;
    if (num < 2) return true;
    return false;
}


#include <stdio.h>

const char* func110(int *lst1, int size1, int *lst2, int size2) {
    int num = 0;
    for (int i = 0; i < size1; i++)
        if (lst1[i] % 2 == 0) num += 1;
    for (int i = 0; i < size2; i++)
        if (lst2[i] % 2 == 0) num += 1;
    if (num >= size1) return "YES";
    return "NO";
}


#include <stdio.h>
#include <string.h>

void func111(const char* test, int* freq, int* max_count, char* letters) {
    int local_freq[26] = {0}; // for 'a' to 'z'
    int local_max = 0;
    const char* ptr = test;
    int idx = 0;

    while (*ptr) {
        if (*ptr != ' ') {
            int letter_index = *ptr - 'a';
            local_freq[letter_index]++;
            if (local_freq[letter_index] > local_max) {
                local_max = local_freq[letter_index];
            }
        }
        ptr++;
    }

    for (int i = 0; i < 26; i++) {
        freq[i] = local_freq[i];
        if (local_freq[i] == local_max) {
            letters[idx++] = 'a' + i;
        }
    }

    *max_count = local_max;
    letters[idx] = '\0';
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

void func112(const char *s, const char *c, char *result, char *palindrome) {
    int len = strlen(s);
    char *n = malloc((len + 1) * sizeof(char));
    int ni = 0;
    for (int i = 0; s[i] != '\0'; i++) {
        const char *temp = c;
        bool found = false;
        while (*temp != '\0') {
            if (s[i] == *temp) {
                found = true;
                break;
            }
            temp++;
        }
        if (!found) {
            n[ni++] = s[i];
        }
    }
    n[ni] = '\0';

    int n_len = strlen(n);
    bool is_palindrome = true;
    for (int i = 0; i < n_len / 2; i++) {
        if (n[i] != n[n_len - 1 - i]) {
            is_palindrome = false;
            break;
        }
    }

    strcpy(result, n);
    strcpy(palindrome, is_palindrome ? "True" : "False");

    free(n);
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char **func113(char *lst[], int size) {
    char **out = malloc(size * sizeof(char *));
    for (int i = 0; i < size; i++) {
        int sum = 0;
        for (int j = 0; lst[i][j] != '\0'; j++) {
            if (lst[i][j] >= '0' && lst[i][j] <= '9' && (lst[i][j] - '0') % 2 == 1)
                sum += 1;
        }
        out[i] = malloc(100); // Assuming the string will not be longer than 99 characters.
        sprintf(out[i], "the number of odd elements %d in the string %d of the %d input.", sum, sum, sum);
    }
    return out;
}


#include <stdio.h>

long long func114(long long *nums, int size) {
    long long current = nums[0];
    long long min = nums[0];
    for (int i = 1; i < size; i++) {
        current = current < 0 ? current + nums[i] : nums[i];
        if (current < min) min = current;
    }
    return min;
}


#include <stdio.h>
int func115(int **grid, int rows, int cols, int capacity) {
    int out = 0;
    for (int i = 0; i < rows; i++) {
        int sum = 0;
        for (int j = 0; j < cols; j++)
            sum += grid[i][j];
        if (sum > 0) out += (sum + capacity - 1) / capacity;
    }
    return out;
}


#include <stdio.h>
#include <stdlib.h>

void func116(int *arr, int size) {
    int count_ones, x, y, temp;
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            count_ones = 0;
            x = arr[i];
            y = arr[j];
            while (x > 0) {
                count_ones += x & 1;
                x >>= 1;
            }
            x = count_ones;
            count_ones = 0;
            while (y > 0) {
                count_ones += y & 1;
                y >>= 1;
            }
            y = count_ones;
            if (y < x || (y == x && arr[j] < arr[i])) {
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

char **func117(const char *s, int n, int *returnSize) {
    const char *vowels = "aeiouAEIOU";
    char **out = NULL;
    int numc = 0, word_count = 0, begin = 0;
    size_t length = strlen(s);
    char *current = (char *)malloc(length + 1);

    for (int i = 0; i <= length; i++) {
        if (isspace(s[i]) || s[i] == '\0') {
            if (numc == n) {
                current[i - begin] = '\0';
                out = (char **)realloc(out, sizeof(char *) * (word_count + 1));
                out[word_count] = (char *)malloc(strlen(current) + 1);
                strcpy(out[word_count], current);
                word_count++;
            }
            begin = i + 1;
            numc = 0;
        } else {
            current[i - begin] = s[i];
            if (strchr(vowels, s[i]) == NULL && isalpha((unsigned char)s[i])) {
                numc++;
            }
        }
    }
    free(current);

    *returnSize = word_count;
    return out;
}


#include <stdio.h>
#include <string.h>

char *func118(const char *word) {
    static char out[2] = {0};
    const char *vowels = "AEIOUaeiou";
    size_t len = strlen(word);
    
    for (int i = len - 2; i >= 1; i--) {
        if (strchr(vowels, word[i]) && !strchr(vowels, word[i + 1]) && !strchr(vowels, word[i - 1])) {
            out[0] = word[i];
            return out;
        }
    }
    out[0] = '\0';
    return out;
}


#include <stdio.h>
#include <string.h>

const char *func119(const char *s1, const char *s2) {
    int count = 0;
    int len1 = strlen(s1);
    int len2 = strlen(s2);
    int i;
    int can = 1;

    for (i = 0; i < len1; i++) {
        if (s1[i] == '(') count++;
        if (s1[i] == ')') count--;
        if (count < 0) can = 0;
    }
    for (i = 0; i < len2; i++) {
        if (s2[i] == '(') count++;
        if (s2[i] == ')') count--;
        if (count < 0) can = 0;
    }
    if (count == 0 && can) return "Yes";

    count = 0;
    can = 1;

    for (i = 0; i < len2; i++) {
        if (s2[i] == '(') count++;
        if (s2[i] == ')') count--;
        if (count < 0) can = 0;
    }
    for (i = 0; i < len1; i++) {
        if (s1[i] == '(') count++;
        if (s1[i] == ')') count--;
        if (count < 0) can = 0;
    }
    if (count == 0 && can) return "Yes";

    return "No";
}


#include <stdio.h>
#include <stdlib.h>

void func120(int* arr, int arr_size, int k, int* out) {
    for (int i = 0; i < arr_size - 1; i++) {
        for (int j = 0; j < arr_size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        out[i] = arr[arr_size - k + i];
    }
}


#include <stdio.h>

int func121(int* lst, int size) {
    int sum = 0;
    for (int i = 0; i * 2 < size; i++)
        if (lst[i * 2] % 2 == 1) sum += lst[i * 2];
    return sum;
}


#include <stdio.h>

int func122(int arr[], int k) {
    int sum = 0;
    for (int i = 0; i < k; i++)
        if (arr[i] >= -99 && arr[i] <= 99)
            sum += arr[i];
    return sum;
}


#include <stdio.h>
#include <stdlib.h>

void func123(int n, int *out, int *size) {
    int capacity = 10;
    *size = 1;
    out[0] = 1;
    
    while (n != 1) {
        if (n % 2 == 1) {
            if (*size >= capacity) {
                capacity *= 2;
                out = (int*)realloc(out, capacity * sizeof(int));
            }
            out[(*size)++] = n;
            n = n * 3 + 1;
        } else {
            n = n / 2;
        }
    }

    for (int i = 1; i < *size; i++) {
        int key = out[i];
        int j = i - 1;

        while (j >= 0 && out[j] > key) {
            out[j + 1] = out[j];
            j = j - 1;
        }
        out[j + 1] = key;
    }
}


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int func124(const char *date) {
    int mm, dd, yy;

    if (strlen(date) != 10) return 0;
    for (int i = 0; i < 10; i++) {
        if (i == 2 || i == 5) {
            if (date[i] != '-') return 0;
        } else {
            if (date[i] < '0' || date[i] > '9') return 0;
        }
    }

    char str_month[3] = {date[0], date[1], '\0'};
    char str_day[3] = {date[3], date[4], '\0'};
    char str_year[5] = {date[6], date[7], date[8], date[9], '\0'};

    mm = atoi(str_month);
    dd = atoi(str_day);
    yy = atoi(str_year);

    if (mm < 1 || mm > 12) return 0;
    if (dd < 1 || dd > 31) return 0;
    if ((mm == 4 || mm == 6 || mm == 9 || mm == 11) && dd == 31) return 0;
    if (mm == 2 && dd > 29) return 0;

    return 1;
}


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

char **func125(const char *txt, int *returnSize) {
    int i, j = 0, num = 0, count = 0;
    int len = strlen(txt);
    char **out = NULL;
    char current[101] = {0};

    if (strchr(txt, ' ') || strchr(txt, ',')) {
        out = malloc(sizeof(char *) * (len + 1));
        for (i = 0; i <= len; ++i) {
            if (txt[i] == ' ' || txt[i] == ',' || txt[i] == '\0') {
                if (j > 0) {
                    current[j] = '\0';
                    out[count] = strdup(current);
                    count++;
                    j = 0;
                }
            } else {
                current[j++] = txt[i];
            }
        }
    } else {
        for (i = 0; i < len; ++i) {
            if (islower(txt[i]) && ((txt[i] - 'a') % 2 == 1)) {
                num++;
            }
        }

        out = malloc(sizeof(char *));
        out[0] = malloc(sizeof(char) * 12);
        sprintf(out[0], "%d", num);
        count = 1;
    }

    *returnSize = count;
    return out;
}


#include <stdio.h>
#include <stdbool.h>

bool func126(const int *lst, int lst_size) {
    if (lst_size == 0) return true;

    for (int i = 1; i < lst_size; i++) {
        if (lst[i] < lst[i - 1]) return false;
        if (i >= 2 && lst[i] == lst[i - 1] && lst[i] == lst[i - 2]) return false;
    }
    return true;
}


#include <stdio.h>

const char* func127(int interval1_start, int interval1_end, int interval2_start, int interval2_end) {
    int inter1, inter2, l, i;
    inter1 = interval1_start > interval2_start ? interval1_start : interval2_start;
    inter2 = interval1_end < interval2_end ? interval1_end : interval2_end;
    l = inter2 - inter1;
    
    if (l < 2) return "NO";
    
    for (i = 2; i * i <= l; i++)
        if (l % i == 0) return "NO";
    
    return "YES";
}


#include <stdio.h>
#include <stdlib.h>

int func128(int *arr, int arr_size) {
    if (arr_size == 0) return -32768;
    int sum = 0, prods = 1, i;
    for (i = 0; i < arr_size; i++) {
        sum += abs(arr[i]);
        if (arr[i] == 0) prods = 0;
        if (arr[i] < 0) prods = -prods;
    }
    return sum * prods;
}


#include <stdio.h>
#include <stdlib.h>

int *func129(int **grid, int N, int k, int *returnSize) {
    int i, j, x, y, min;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            if (grid[i][j] == 1) {
                x = i;
                y = j;
            }
    min = N * N;
    if (x > 0 && grid[x - 1][y] < min) min = grid[x - 1][y];
    if (x < N - 1 && grid[x + 1][y] < min) min = grid[x + 1][y];
    if (y > 0 && grid[x][y - 1] < min) min = grid[x][y - 1];
    if (y < N - 1 && grid[x][y + 1] < min) min = grid[x][y + 1];
    
    *returnSize = k;
    int *out = (int *)malloc(k * sizeof(int));
    for (i = 0; i < k; i++)
        if (i % 2 == 0) out[i] = 1;
        else out[i] = min;
    return out;
}


#include <stdio.h>
#include <stdlib.h>

int* func130(int n) {
    int* out = (int*)malloc((n + 1) * sizeof(int));
    out[0] = 1;
    if (n == 0) return out;
    out[1] = 3;
    for (int i = 2; i <= n; i++) {
        if (i % 2 == 0) out[i] = 1 + i / 2;
        else out[i] = out[i - 1] + out[i - 2] + 1 + (i + 1) / 2;
    }
    return out;
}


#include <stdio.h>

int func131(int n) {
    int prod = 1, has_odd = 0, digit;
    while (n > 0) {
        digit = n % 10;
        if (digit % 2 == 1) {
            has_odd = 1;
            prod *= digit;
        }
        n /= 10;
    }
    return has_odd ? prod : 0;
}


#include <stdio.h>
#include <string.h>

int func132(const char *str) {
    int count = 0, maxcount = 0;
    for (int i = 0; i < strlen(str); i++) {
        if (str[i] == '[') count += 1;
        if (str[i] == ']') count -= 1;
        if (count < 0) count = 0;
        if (count > maxcount) maxcount = count;
        if (count <= maxcount - 2) return 1;
    }
    return 0;
}


#include <stdio.h>
#include <math.h>

int func133(float *lst, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += (int)ceil(lst[i]) * (int)ceil(lst[i]);
    }
    return sum;
}


#include <stdio.h>
#include <string.h>
#include <ctype.h>

int func134(const char *txt) {
    int len = strlen(txt);
    if (len == 0) return 0;
    char last_char = txt[len - 1];
    if (!isalpha((unsigned char)last_char)) return 0;
    if (len == 1) return 1;
    char second_last_char = txt[len - 2];
    if (isalpha((unsigned char)second_last_char)) return 0;
    return 1;
}


#include <stdio.h>

int func135(int *arr, int size) {
    int max = -1;
    for (int i = 1; i < size; ++i) {
        if (arr[i] < arr[i - 1]) max = i;
    }
    return max;
}


#include <stdio.h>

void func136(const int *lst, int size, int result[2]) {
    int maxneg = 0;
    int minpos = 0;
    for (int i = 0; i < size; i++) {
        if (lst[i] < 0 && (maxneg == 0 || lst[i] > maxneg)) maxneg = lst[i];
        if (lst[i] > 0 && (minpos == 0 || lst[i] < minpos)) minpos = lst[i];
    }
    result[0] = maxneg;
    result[1] = minpos;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* func137(const char* a, const char* b) {
    char *end;
    char *a_modified = strdup(a);
    char *b_modified = strdup(b);

    // Replace ',' with '.' if present for atof conversion
    for (int i = 0; a_modified[i]; ++i) if (a_modified[i] == ',') a_modified[i] = '.';
    for (int i = 0; b_modified[i]; ++i) if (b_modified[i] == ',') b_modified[i] = '.';

    double numa = strtod(a_modified, &end);
    if (*end) { free(a_modified); free(b_modified); return "Invalid input"; } // Not a valid number
    double numb = strtod(b_modified, &end);
    if (*end) { free(a_modified); free(b_modified); return "Invalid input"; } // Not a valid number

    free(a_modified);
    free(b_modified);

    if (numa == numb) return "None";
    return numa > numb ? (char*)a : (char*)b;
}


#include <stdio.h>

int func138(int n) {
    if (n % 2 == 0 && n >= 8) return 1;
    return 0;
}


#include <stdio.h>

long long func139(int n) {
    long long fact = 1, bfact = 1;
    for (int i = 1; i <= n; i++) {
        fact = fact * i;
        bfact = bfact * fact;
    }
    return bfact;
}


#include <stdio.h>
#include <string.h>

void func140(const char *text, char *out) {
    int space_len = 0;
    int j = 0;
    for (int i = 0; i < strlen(text); i++) {
        if (text[i] == ' ') {
            space_len++;
        } else {
            if (space_len == 1) out[j++] = '_';
            if (space_len == 2) out[j++] = '_', out[j++] = '_';
            if (space_len > 2) out[j++] = '-';
            space_len = 0;
            out[j++] = text[i];
        }
    }
    if (space_len == 1) out[j++] = '_';
    if (space_len == 2) out[j++] = '_', out[j++] = '_';
    if (space_len > 2) out[j++] = '-';
    out[j] = '\0';
}


#include <stdio.h>
#include <string.h>

const char* func141(const char* file_name) {
    int num_digit = 0, num_dot = 0;
    int length = strlen(file_name);
    if (length < 5) return "No";
    char w = file_name[0];
    if (w < 'A' || (w > 'Z' && w < 'a') || w > 'z') return "No";
    const char* last = file_name + length - 4;
    if (strcmp(last, ".txt") != 0 && strcmp(last, ".exe") != 0 && strcmp(last, ".dll") != 0) return "No";
    for (int i = 0; i < length; i++) {
        if (file_name[i] >= '0' && file_name[i] <= '9') num_digit++;
        if (file_name[i] == '.') num_dot++;
    }
    if (num_digit > 3 || num_dot != 1) return "No";
    return "Yes";
}


#include <stdio.h>

int func142(int* lst, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        if (i % 3 == 0) sum += lst[i] * lst[i];
        else if (i % 4 == 0) sum += lst[i] * lst[i] * lst[i];
        else sum += lst[i];
    }
    return sum;
}


#include <stdio.h>
#include <string.h>
#include <stdbool.h>

void func143(const char* sentence, char* out) {
    int index = 0, word_len = 0;
    int out_index = 0;
    bool is_prime;
    int i, j;

    for (i = 0; sentence[i] != '\0'; ++i) {
        if (sentence[i] != ' ') {
            word_len++;
        } else {
            if (word_len > 1) {
                is_prime = true;
                for (j = 2; j * j <= word_len; ++j) {
                    if (word_len % j == 0) {
                        is_prime = false;
                        break;
                    }
                }
            } else {
                is_prime = false;
            }

            if (is_prime) {
                if (out_index > 0) {
                    out[out_index++] = ' ';
                }
                memcpy(out + out_index, sentence + i - word_len, word_len);
                out_index += word_len;
            }
            word_len = 0;
        }
    }

    if (word_len > 1) {
        is_prime = true;
        for (j = 2; j * j <= word_len; ++j) {
            if (word_len % j == 0) {
                is_prime = false;
                break;
            }
        }
    } else {
        is_prime = false;
    }

    if (is_prime) {
        if (out_index > 0) {
            out[out_index++] = ' ';
        }
        memcpy(out + out_index, sentence + i - word_len, word_len);
        out_index += word_len;
    }

    out[out_index] = '\0';
}


#include <stdio.h>
#include <stdlib.h>

int func144(const char* x, const char* n){
    int a, b, c, d, i, j;
    char num[101], den[101];

    for (i = 0; x[i] != '/'; i++) {
        num[i] = x[i];
    }
    num[i] = '\0';
    a = atoi(num);

    for (j = 0, i = i + 1; x[i] != '\0'; i++, j++) {
        den[j] = x[i];
    }
    den[j] = '\0';
    b = atoi(den);

    for (i = 0; n[i] != '/'; i++) {
        num[i] = n[i];
    }
    num[i] = '\0';
    c = atoi(num);

    for (j = 0, i = i + 1; n[i] != '\0'; i++, j++) {
        den[j] = n[i];
    }
    den[j] = '\0';
    d = atoi(den);

    if ((a * c) % (b * d) == 0) return 1;
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int* func145(int nums[], int size) {
    int* sumdigit = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        char w[12]; // Assuming the integer won't exceed the length of an int in string form.
        sprintf(w, "%d", abs(nums[i]));
        int sum = 0, length = strlen(w);
        for (int j = 1; j < length; j++)
            sum += w[j] - '0';
        if (nums[i] > 0) sum += w[0] - '0';
        else sum -= w[0] - '0';
        sumdigit[i] = sum;
    }
    int m;
    for (int i = 0; i < size; i++)
        for (int j = 1; j < size; j++)
            if (sumdigit[j - 1] > sumdigit[j]) {
                m = sumdigit[j]; sumdigit[j] = sumdigit[j - 1]; sumdigit[j - 1] = m;
                m = nums[j]; nums[j] = nums[j - 1]; nums[j - 1] = m;
            }
    
    free(sumdigit);
    return nums;
}


#include <stdio.h>
#include <stdlib.h>

int func146(int nums[], int size) {
    int num = 0;
    for (int i = 0; i < size; i++) {
        if (nums[i] > 10) {
            int first, last;
            last = nums[i] % 10;
            int n = nums[i];
            while (n >= 10) {
                n /= 10;
            }
            first = n;
            if (first % 2 == 1 && last % 2 == 1) {
                num += 1;
            }
        }
    }
    return num;
}


#include <stdio.h>
#include <stdlib.h>

int func147(int n) {
    int *a = (int *)malloc(n * sizeof(int));
    int **sum = (int **)malloc((n + 1) * sizeof(int *));
    int **sum2 = (int **)malloc((n + 1) * sizeof(int *));
    for (int i = 0; i <= n; i++) {
        sum[i] = (int *)calloc(3, sizeof(int));
        sum2[i] = (int *)calloc(3, sizeof(int));
    }
    sum[0][0] = sum[0][1] = sum[0][2] = 0;
    sum2[0][0] = sum2[0][1] = sum2[0][2] = 0;
    for (int i = 1; i <= n; i++) {
        a[i - 1] = (i * i - i + 1) % 3;
        for (int j = 0; j < 3; j++) {
            sum[i][j] = sum[i - 1][j];
        }
        sum[i][a[i - 1]] += 1;
    }
    for (int times = 1; times < 3; times++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < 3; j++) {
                sum2[i][j] = sum2[i - 1][j];
            }
            if (i >= 1) {
                for (int j = 0; j <= 2; j++) {
                    sum2[i][(a[i - 1] + j) % 3] += sum[i - 1][j];
                }
            }
        }
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j < 3; j++) {
                sum[i][j] = sum2[i][j];
                sum2[i][j] = 0;
            }
        }
    }

    int result = sum[n][0];
    for (int i = 0; i <= n; ++i) {
        free(sum[i]);
        free(sum2[i]);
    }
    free(sum);
    free(sum2);
    free(a);
    return result;
}


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char** func148(const char* planet1, const char* planet2, int* returnSize) {
    const char* planets[] = {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"};
    int pos1 = -1, pos2 = -1, m;
    for (m = 0; m < 8; m++) {
        if (strcmp(planets[m], planet1) == 0) pos1 = m;
        if (strcmp(planets[m], planet2) == 0) pos2 = m;
    }
    if (pos1 == -1 || pos2 == -1 || pos1 == pos2) {
        *returnSize = 0;
        return NULL;
    }
    if (pos1 > pos2) { int temp = pos1; pos1 = pos2; pos2 = temp; }
    *returnSize = pos2 - pos1 - 1;
    if (*returnSize <= 0) {
        *returnSize = 0;
        return NULL;
    }
    char** out = malloc(*returnSize * sizeof(char*));
    for (m = pos1 + 1; m < pos2; m++) {
        out[m - pos1 - 1] = (char*)planets[m];
    }
    return out;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char** func149(char **lst, int lst_size, int *return_size) {
    int i, j;
    char *temp;

    *return_size = 0;
    for (i = 0; i < lst_size; ++i) {
        if (strlen(lst[i]) % 2 == 0) {
            lst[*return_size] = lst[i];
            (*return_size)++;
        }
    }

    for (i = 0; i < *return_size - 1; ++i) {
        for (j = 0; j < *return_size - i - 1; ++j) {
            size_t len_j = strlen(lst[j]);
            size_t len_j1 = strlen(lst[j + 1]);
            if (len_j > len_j1 || (len_j == len_j1 && strcmp(lst[j], lst[j + 1]) > 0)) {
                temp = lst[j];
                lst[j] = lst[j + 1];
                lst[j + 1] = temp;
            }
        }
    }

    char **out = malloc(*return_size * sizeof(char *));
    for (i = 0; i < *return_size; ++i) {
        out[i] = lst[i];
    }

    return out;
}


#include <stdio.h>

int func150(int n, int x, int y) {
    int isp = 1;
    if (n < 2) isp = 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) isp = 0;
    }
    if (isp) return x;
    return y;
}


#include <stdio.h>
#include <math.h>

long long func151(float lst[], int lst_size) {
    long long sum = 0;
    for (int i = 0; i < lst_size; i++) {
        if (fabs(lst[i] - round(lst[i])) < 1e-4) {
            if (lst[i] > 0 && (int)(round(lst[i])) % 2 == 1) {
                sum += (int)(round(lst[i])) * (int)(round(lst[i]));
            }
        }
    }
    return sum;
}


#include <stdio.h>
#include <stdlib.h>

void func152(int* game, int* guess, int* out, int length) {
    for (int i = 0; i < length; i++) {
        out[i] = abs(game[i] - guess[i]);
    }
}


#include <stdio.h>
#include <string.h>

void func153(const char* class_name, const char** extensions, int ext_count, char* output) {
    int max_strength = -1000;
    const char* strongest = NULL;
    for (int i = 0; i < ext_count; i++) {
        const char* extension = extensions[i];
        int strength = 0;
        for (int j = 0; extension[j] != '\0'; j++) {
            char chr = extension[j];
            if (chr >= 'A' && chr <= 'Z') strength++;
            if (chr >= 'a' && chr <= 'z') strength--;
        }
        if (strength > max_strength) {
            max_strength = strength;
            strongest = extension;
        }
    }
    sprintf(output, "%s.%s", class_name, strongest);
}


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

bool func154(const char *a, const char *b) {
    int len_a = strlen(a);
    int len_b = strlen(b);
    char *temp = (char *)malloc(2 * len_b + 1);

    for (int i = 0; i < len_b; i++) {
        strncpy(temp, b + i, len_b - i);
        strncpy(temp + len_b - i, b, i);
        temp[len_b] = '\0';
        if (strstr(a, temp)) {
            free(temp);
            return true;
        }
    }

    free(temp);
    return false;
}


#include <stdio.h>
#include <stdlib.h>

void func155(int num, int *result) {
    int even_count = 0, odd_count = 0;
    num = abs(num);
    
    do {
        int digit = num % 10;
        if (digit % 2 == 0) {
            even_count++;
        } else {
            odd_count++;
        }
        num /= 10;
    } while (num > 0);

    result[0] = even_count;
    result[1] = odd_count;
}


#include <stdio.h>
#include <string.h>

void func156(int number, char *result) {
    const char *rep[] = {"m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i"};
    const int num[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    int pos = 0;
    result[0] = '\0';

    while(number > 0) {
        while (number >= num[pos]) {
            strcat(result, rep[pos]);
            number -= num[pos];
        }
        if (number > 0) pos++;
    }
}


#include <stdio.h>
#include <math.h>

int func157(float a, float b, float c) {
    if (fabs(a*a + b*b - c*c) < 1e-4 || fabs(a*a + c*c - b*b) < 1e-4 || fabs(b*b + c*c - a*a) < 1e-4) return 1;
    return 0;
}


#include <stdio.h>
#include <string.h>

char *func158(char *words[], int count) {
    char *max = "";
    int maxu = 0;
    for (int i = 0; i < count; i++) {
        char unique[256] = {0};
        int unique_count = 0;
        for (int j = 0; words[i][j] != '\0'; j++) {
            if (!strchr(unique, words[i][j])) {
                int len = strlen(unique);
                unique[len] = words[i][j];
                unique[len + 1] = '\0';
                unique_count++;
            }
        }
        if (unique_count > maxu || (unique_count == maxu && strcmp(words[i], max) < 0)) {
            max = words[i];
            maxu = unique_count;
        }
    }
    return max;
}


#include <stdio.h>

void func159(int number, int need, int remaining, int result[2]) {
    if (need > remaining) {
        result[0] = number + remaining;
        result[1] = 0;
    } else {
        result[0] = number + need;
        result[1] = remaining - need;
    }
}


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int func160(const char **operato, const int *operand, int operato_size, int operand_size) {
    int *num = (int*)malloc(operand_size * sizeof(int));
    int *posto = (int*)malloc(operand_size * sizeof(int));
    for (int i = 0; i < operand_size; i++) {
        num[i] = operand[i];
        posto[i] = i;
    }
    for (int i = 0; i < operato_size; i++) {
        if (strcmp(operato[i], "**") == 0) {
            while (posto[posto[i]] != posto[i]) posto[i] = posto[posto[i]];
            while (posto[posto[i + 1]] != posto[i + 1]) posto[i + 1] = posto[posto[i + 1]];
            num[posto[i]] = pow(num[posto[i]], num[posto[i + 1]]);
            posto[i + 1] = posto[i];
        }
    }
    for (int i = 0; i < operato_size; i++) {
        if (strcmp(operato[i], "*") == 0 || strcmp(operato[i], "//") == 0) {
            while (posto[posto[i]] != posto[i]) posto[i] = posto[posto[i]];
            while (posto[posto[i + 1]] != posto[i + 1]) posto[i + 1] = posto[posto[i + 1]];
            if (strcmp(operato[i], "*") == 0)
                num[posto[i]] *= num[posto[i + 1]];
            else
                num[posto[i]] /= num[posto[i + 1]];
            posto[i + 1] = posto[i];
        }
    }
    for (int i = 0; i < operato_size; i++) {
        if (strcmp(operato[i], "+") == 0 || strcmp(operato[i], "-") == 0) {
            while (posto[posto[i]] != posto[i]) posto[i] = posto[posto[i]];
            while (posto[posto[i + 1]] != posto[i + 1]) posto[i + 1] = posto[posto[i + 1]];
            if (strcmp(operato[i], "+") == 0)
                num[posto[i]] += num[posto[i + 1]];
            else
                num[posto[i]] -= num[posto[i + 1]];
            posto[i + 1] = posto[i];
        }
    }
    int result = num[0];
    free(num);
    free(posto);
    return result;
}


#include <stdio.h>
#include <string.h>
#include <ctype.h>

char* func161(char *s){
    int nletter = 0;
    int length = strlen(s);
    for (int i = 0; i < length; i++) {
        if (isalpha((unsigned char)s[i])) {
            if (isupper((unsigned char)s[i])) s[i] = tolower((unsigned char)s[i]);
            else if (islower((unsigned char)s[i])) s[i] = toupper((unsigned char)s[i]);
        } else {
            nletter += 1;
        }
    }
    if (nletter == length) {
        for (int i = 0; i < length / 2; i++) {
            char temp = s[i];
            s[i] = s[length - i - 1];
            s[length - i - 1] = temp;
        }
    }
    return s;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CRC32 function
char* func162(const char* text) {
    if(strlen(text) == 0) {
        return strdup("None");
    }

    unsigned int crc = 0xFFFFFFFF; 
    unsigned int i, j;
    unsigned char byte;

    for(i = 0; text[i] != '\0'; i++) {
        byte = text[i];
        crc = crc ^ byte;
        for(j = 0; j < 8; j++) {
            if(crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc = crc >> 1;
            }
        }
    }
    crc = ~crc;

    char* result = malloc(9);
    if(result != NULL) {
        sprintf(result, "%08X", crc);
    }

    return result;
}


#include <stdio.h>

void func163(int a, int b, int *out, int *size) {
    int m;
    *size = 0;

    if (b < a) {
        m = a;
        a = b;
        b = m;
    }

    for (int i = a; i <= b; i++) {
        if (i < 10 && i % 2 == 0) {
            out[(*size)++] = i;
        }
    }
}