#include <stdio.h>

int main(void){ 
    char array[100];

    snprintf(array, sizeof(array), "yay.txt");
    
    FILE *file = fopen(array, "w");
    
    if (file == NULL) {
        perror("Error opening file");
        return 1;  
    }

    fprintf(file, "Hello, this is me Paris!\n");
    fclose(file);

}