#pragma once
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>


void image_resize(const char* sFilename, float ** image_h, int* original_image_h, int* original_image_w, int resized_h, int resized_w);

void draw_box(const char* sFilename, float *locData, int box_address, int box_total);