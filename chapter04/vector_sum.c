/* #include "../common/book.h" */
#include <stdio.h>

#define N 10

void add(long int *a, long int *b, long int *c)
{
    long int tid = 0;    // this is CPU zero so we start at zero
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += 1; // we have one CPU so we increment by one
    }
}

int main(int argc, char *argv[])
{
    long int a[N], b[N], c[N];

    // fill the arrays a and b on the CPU
    for (long int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * i;
    }

    add(a, b, c);

    for (int i = 0; i < N; ++i) {
        printf("%ld + %ld = %ld\n", a[i], b[i], c[i]);
    }
    return 0;
}
