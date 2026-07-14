#include <stdio.h>


struct MyStruct {
    int number1;
    float number2;
};

int main(void) { 
    struct MyStruct myStruct;
    myStruct.number1 = 10;
    myStruct.number2 = 3.14;

    printf("Number 1: %d\n", myStruct.number1);
    printf("Number 2: %.2f\n", myStruct.number2);

    char filename[100];
    snprintf(filename, sizeof(filename), "snapshot.txt");
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }
    fprintf(fp, "Hello, World!");
    fclose(fp);

    printf("Filename: %s\n", pt);


    // snprintf(*filename, sizeof(*filename), "snapshot.txt");
    return 0;
}