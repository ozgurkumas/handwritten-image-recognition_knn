#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define TRAIN_IMAGE_FILE "train-images-idx3-ubyte"
#define TRAIN_LABEL_FILE "train-labels-idx1-ubyte"
#define IMAGE_SIZE 28 * 28
#define LABEL_SIZE 1
#define K 3

typedef struct {
    uint8_t *image;
    uint8_t label;
} Sample;

float distance(uint8_t list1[], uint8_t list2[], int length) {
    float dist = 0;
    for (int i = 0; i < length; i++) {
        dist += pow(list1[i] - list2[i], 2);
    }
    return sqrt(dist);
}

int maxCount(int list[], int num) {
    int freq[10] = {0};
    for (int i = 0; i < num; i++) {
        freq[list[i]]++;
    }
    int max = 0;
    int index = 0;
    for (int i = 0; i < 10; i++) {
        if (max < freq[i]) {
            max = freq[i];
            index = i;
        }
    }
    return index;
}

void kNearestNeighbors(uint8_t train[][784], uint8_t label[], uint8_t test[], int num, int k, int *neighborList) {
    typedef struct {
        float distance;
        int label;
    } DistanceLabel;

    DistanceLabel *distances = malloc(sizeof(DistanceLabel) * num);

    for (int i = 0; i < num; i++) {
        distances[i].distance = distance(train[i], test, 784);
        distances[i].label = label[i];
    }

    for (int i = 0; i < num - 1; i++) {
        for (int j = 0; j < num - i - 1; j++) {
            if (distances[j].distance > distances[j + 1].distance) {
                DistanceLabel temp = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = temp;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        neighborList[i] = distances[i].label;
    }

    free(distances);
}

int main() {
    FILE *imageFile, *labelFile;
    uint8_t *images, *labels;
    uint32_t magicNumber, numImages, numRows, numCols, i;

    imageFile = fopen(TRAIN_IMAGE_FILE, "rb");
    if (!imageFile) {
        fprintf(stderr, "Resim dosyası açılamadı.\n");
        return 1;
    }

    labelFile = fopen(TRAIN_LABEL_FILE, "rb");
    if (!labelFile) {
        fprintf(stderr, "Etiket dosyası açılamadı.\n");
        fclose(imageFile);
        return 1;
    }

    fread(&magicNumber, sizeof(uint32_t), 1, labelFile);
    fread(&numImages, sizeof(uint32_t), 1, labelFile);
    fread(&numRows, sizeof(uint32_t), 1, imageFile);
    fread(&numCols, sizeof(uint32_t), 1, imageFile);

    images = (uint8_t *)malloc(IMAGE_SIZE * numImages);
    labels = (uint8_t *)malloc(LABEL_SIZE * numImages);

    fread(images, sizeof(uint8_t), IMAGE_SIZE * numImages, imageFile);
    fread(labels, sizeof(uint8_t), LABEL_SIZE * numImages, labelFile);

    Sample *dataset = (Sample *)malloc(sizeof(Sample) * numImages);

    for (i = 0; i < numImages; ++i) {
        dataset[i].image = images + (i * IMAGE_SIZE);
        dataset[i].label = labels[i];
    }

    uint8_t train[2000][784];
    uint8_t label_train[2000];

    for (i = 0; i < 2000; i++) {
        for (int j = 0; j < 784; j++) {
            train[i][j] = dataset[i].image[j];
        }
        label_train[i] = dataset[i].label;
    }

    int accurate = 0;
    for(int i=2000; i<4000; i++)
    {
        int neighborList[K];
        kNearestNeighbors(train, label_train, dataset[i].image, 2000, K, neighborList);
        int prediction = maxCount(neighborList, K);
        printf("index: %d tahmin: %d istenen: %d\n", i-2000, prediction, dataset[i].label);
        if(prediction==dataset[i].label)
        {
            accurate += 1;
        }
    }
    float accuracy =  ((float) accurate)/2000;
    printf("%f", accuracy);

    free(images);
    free(labels);
    free(dataset);

    fclose(imageFile);
    fclose(labelFile);

    return 0;
}
