///////////////////////////////////////////////////////////////////////////////
//
//      TargaImage.cpp                          Author:     Stephen Chenney
//                                              Modified:   Eric McDaniel
//                                              Date:       Fall 2004
//                                              Modified:   Feng Liu
//                                              Date:       Winter 2011
//                                              Why:        Change the library file 
//      Implementation of TargaImage methods.  You must implement the image
//  modification functions.
//
///////////////////////////////////////////////////////////////////////////////

#include "Globals.h"
#include "TargaImage.h"
#include "libtarga.h"
#include <stdlib.h>
#include <assert.h>
#include <unordered_map>
#include <memory.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <valarray>
#include <vector>
#include <functional>
#include <algorithm>

using namespace std;

// constants
const int           RED             = 0;                // red channel
const int           GREEN           = 1;                // green channel
const int           BLUE            = 2;  // blue channel
const unsigned char BACKGROUND[3]   = { 0, 0, 0 };      // background color


// Computes n choose s, efficiently
double Binomial(int n, int s)
{
    double        res;

    res = 1;
    for (int i = 1 ; i <= s ; i++)
        res = (n - i + 1) * res / i ;

    return res;
}// Binomial


///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage() : width(0), height(0), data(NULL)
{}// TargaImage

///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(int w, int h) : width(w), height(h)
{
   data = new unsigned char[width * height * 4];
   ClearToBlack();
}// TargaImage



///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables to values given.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(int w, int h, unsigned char *d)
{
    int i;

    width = w;
    height = h;
    data = new unsigned char[width * height * 4];

    for (i = 0; i < width * height * 4; i++)
	    data[i] = d[i];
}// TargaImage

///////////////////////////////////////////////////////////////////////////////
//
//      Copy Constructor.  Initialize member to that of input
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(const TargaImage& image) 
{
   width = image.width;
   height = image.height;
   data = NULL; 
   if (image.data != NULL) {
      data = new unsigned char[width * height * 4];
      memcpy(data, image.data, sizeof(unsigned char) * width * height * 4);
   }
}


///////////////////////////////////////////////////////////////////////////////
//
//      Destructor.  Free image memory.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::~TargaImage()
{
    if (data)
        delete[] data;
}// ~TargaImage


///////////////////////////////////////////////////////////////////////////////
//
//      Converts an image to RGB form, and returns the rgb pixel data - 24 
//  bits per pixel. The returned space should be deleted when no longer 
//  required.
//
///////////////////////////////////////////////////////////////////////////////
unsigned char* TargaImage::To_RGB(void)
{
    unsigned char   *rgb = new unsigned char[width * height * 3];
    int		    i, j;

    if (! data)
	    return NULL;

    // Divide out the alpha
    for (i = 0 ; i < height ; i++)
    {
	    int in_offset = i * width * 4;
	    int out_offset = i * width * 3;

	    for (j = 0 ; j < width ; j++)
        {
	        RGBA_To_RGB(data + (in_offset + j*4), rgb + (out_offset + j*3));
	    }
    }

    return rgb;
}// TargaImage


///////////////////////////////////////////////////////////////////////////////
//
//      Save the image to a targa file. Returns 1 on success, 0 on failure.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Save_Image(const char *filename)
{
    TargaImage	*out_image = Reverse_Rows();

    if (! out_image)
	    return false;

    if (!tga_write_raw(filename, width, height, out_image->data, TGA_TRUECOLOR_32))
    {
	    cout << "TGA Save Error: %s\n", tga_error_string(tga_get_last_error());
	    return false;
    }

    delete out_image;

    return true;
}// Save_Image


///////////////////////////////////////////////////////////////////////////////
//
//      Load a targa image from a file.  Return a new TargaImage object which 
//  must be deleted by caller.  Return NULL on failure.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage* TargaImage::Load_Image(char *filename)
{
    unsigned char   *temp_data;
    TargaImage	    *temp_image;
    TargaImage	    *result;
    int		        width, height;

    if (!filename)
    {
        cout << "No filename given." << endl;
        return NULL;
    }// if

    temp_data = (unsigned char*)tga_load(filename, &width, &height, TGA_TRUECOLOR_32);
    if (!temp_data)
    {
        cout << "TGA Error: %s\n", tga_error_string(tga_get_last_error());
	    width = height = 0;
	    return NULL;
    }
    temp_image = new TargaImage(width, height, temp_data);
    free(temp_data);

    result = temp_image->Reverse_Rows();

    delete temp_image;

    return result;
}// Load_Image


///////////////////////////////////////////////////////////////////////////////
//
//      Convert image to grayscale.  Red, green, and blue channels should all 
//  contain grayscale value.  alpha channel shoould be left unchanged.  Return
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::To_Grayscale()
{
    if (!data)
        return false;
    const int canvasSize = width* height * 4;

    for (int i = 0; i < canvasSize; i += 4)
    {
        unsigned char rgb[3];
        unsigned char rgb_to_gray;

        RGBA_To_RGB(data + i, rgb);
        
        rgb_to_gray = (unsigned char)(((double)rgb[0]*.299) + ((double)rgb[1]*.587) + ((double)rgb[2]*.114));
        for (int j = 0; j < 3; j++)
            data[i + j]= rgb_to_gray;
    }

    return true;
}// To_Grayscale


///////////////////////////////////////////////////////////////////////////////
//
//  Convert the image to an 8 bit image using uniform quantization.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Quant_Uniform()
{
    if (!data)
        return false;
    const int canvasSize = width * height * 4;

    for (int i = 0; i < canvasSize; i += 4)
    {
        unsigned char rgb[3];

        RGBA_To_RGB(data + i, rgb);
        data[i + 0] = rgb[0] & (~((1 << 5) - 1));
        data[i + 1] = rgb[1] & (~((1 << 5) - 1));
        data[i + 2] = rgb[2] & (~((1 << 6) - 1));
    }

    return true;
}// Quant_Uniform


///////////////////////////////////////////////////////////////////////////////
//
//      Convert the image to an 8 bit image using populosity quantization.  
//  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////

typedef tuple<int, int, int> key_t;
struct key_hash : public unary_function<key_t, size_t> {
    size_t operator()(const key_t &k) const {
        return (get<0>(k) * 257) + get<1>(k) + get<2>(k);
    }
};
bool TargaImage::Quant_Populosity()
{
    if (!data) {
        return false;
    }
    const int canvasSize = width * height * 4;
    std::unordered_map<key_t, int, key_hash> map_t;
    const int swatches = 256;
    const float range = (float)swatches/32.0;
    for (int i = 0; i < canvasSize; i += 4){
        for (int j = 0; j < 3; j++)
            data[i + j] = round(floor(data[i + j] / range)) * range;
        map_t[make_tuple(data[i], data[i + 1], data[i + 2])] += 1;
    }

    vector<pair<key_t, int>> chart;
    vector<key_t> pop_swatch;

    for (auto temp = map_t.begin(); temp != map_t.end(); ++temp) {
        chart.push_back(make_pair(temp->first, temp->second));
    }

    sort(chart.begin(), chart.end(), [](auto &left, auto &right) {
        return left.second > right.second;
        });

    for (int i = 0; i < swatches; i++) {
        pop_swatch.push_back(chart[i].first);
    }

    for (int i = 0; i < canvasSize; i += 4) {

        key_t temp = make_tuple(data[i], data[i + 1], data[i + 2]);
        key_t match = *min_element(pop_swatch.begin(), pop_swatch.end(),
            [&temp, this](const auto &left, const auto &right) {
                int dlx = (float)get<0>(temp) - get<0>(left);
                int dly = (float)get<1>(temp) - get<1>(left);
                int dlz = (float)get<2>(temp) - get<2>(left);
                int dSqL = (dlx * dlx) + (dly * dly) + (dlz * dlz);
                int drx = (float)get<0>(temp) - get<0>(right);
                int dry = (float)get<1>(temp) - get<1>(right);
                int drz = (float)get<2>(temp) - get<2>(right);
                return dSqL < ((drx * drx) + (dry * dry) + (drz * drz));
            }
        );
            data[i] = (float)get<0>(match);
            data[i + 1] = (float)get<1>(match);
            data[i + 2] = (float)get<2>(match);
    }

    return true;
}// Quant_Populosity


///////////////////////////////////////////////////////////////////////////////
//
//      Dither the image using a threshold of 1/2.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Threshold()
{
    if (!data)
        return false;
    const int canvasSize = width * height * 4;
    To_Grayscale();

    for (int i = 0; i < canvasSize; i += 4)
    {
        unsigned char rgb[3];
        unsigned char threshold;

        RGBA_To_RGB(data + i, rgb);

        if (rgb[0] > 128)
            threshold = 255;
        else
            threshold = 0;

        for (int j = 0; j < 3; j++)
            data[i + j] = threshold;
    }
    return true;
}// Dither_Threshold


///////////////////////////////////////////////////////////////////////////////
//
//      Dither image using random dithering.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Random()
{
    srand(time(NULL));

    if (!data)
        return false;

    To_Grayscale();
    const int canvasSize = width * height * 4;

    int total = 0;
    for (int i = 0; i < canvasSize; i += 4)
        total += data[i];
    total /= (canvasSize / 4);

    for (int i = 0; i < canvasSize; i += 4)
    {
        unsigned char rgb[3];
        unsigned char threshold = 0;
        int randX;
        int randY;

        RGBA_To_RGB(data + i, rgb);
        randX = (rand() % 52) - 26;
        randY = randX + (int)rgb[0];

        if (rgb[0] > 128)
            threshold = 255;

        for (int j = 0; j < 3; j++)
            data[i + j] = threshold;
    }
    return true;
}// Dither_Random


///////////////////////////////////////////////////////////////////////////////
//
//      Perform Floyd-Steinberg dithering on the image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////


bool TargaImage::Dither_FS()
{
    if (!data)
        return false;

    To_Grayscale();
 #define vertex(a,b) (data + (width * b * 4) + (a * 4))
    int start = width - 1;
    int direction = 1;
    int count = 0;

    for (int j = 0; j < height; j++) {
        for (int i = count; (direction == 1) ? (i <= start) : (i >= start); i += direction) {
            double d;
            unsigned char threshold = 0;
            
            unsigned char s = (double)*(vertex(i, j));

            if(s > 128)
                threshold = 255.0;

            d = s - threshold;

            for (int l = 0; l < 3; l++)
                *(vertex(i, j) + l) = threshold;
            *(vertex(i, j) + 3) = 255;

            int x;
            int y;
            double z;
            for (int k = 0; k < 4; k++) {
                switch (k) {
                case 0:
                    x = i + direction;
                    y = j + 0;
                    z = (7.0 * d) / 16.0;
                    break;
                case 1:
                    x = i - direction;
                    y = j + 1;
                    z = (3.0 * d) / 16.0;
                    break;
                case 2:
                    x = i;
                    y = j + 1;
                    z = (5.0 * d) / 16.0;
                    break;
                case 3:
                    x = i + direction;
                    y = j + 1;
                    z = (1.0 * d) / 16.0;
                }

                double q;

                if (x >= 0 && x < width) {
                    if (y >= 0 && y < height) {
                        q = *(vertex(x, y));
                        q += z;
                        for (int p = 0; p < 3; p++)
                            *(vertex(x, y) + p) = q;
                        *(vertex(x, y) + 3) = 255;
                    }
                }
            }
        }
        int temp = start;
        start = count;
        count = temp;
        direction = -direction;
    }
    return true;
}// Dither_FS


///////////////////////////////////////////////////////////////////////////////
//
//      Dither the image while conserving the average brightness.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Bright()
{
    if (!data)
        return false;
    vector<unsigned char> v;
    int start;
    double pp;
    double num;
    const int canvasSize = width * height * 4;

    To_Grayscale();

    signed int total = 0;
    for (int i = 0; i < canvasSize; i += 4) {
        total += data[i];
        v.push_back(data[i]);
    }
    total /= (width * height);
    sort(v.begin(), v.end());

    pp = (double)total / 255.0;
    num = v.size();

    start = (1.0 - pp) * num;

    for (int i = 0; i < canvasSize; i += 4) {
        unsigned char rgb[3];
        unsigned char threshold = 0;

        RGBA_To_RGB(data + i, rgb);

        if (rgb[0] >= v[start])
            threshold = 255;

        for (int j = 0; j < 3; j++)
            data[i + j] = threshold;
    }
    return true;
}// Dither_Bright


///////////////////////////////////////////////////////////////////////////////
//
//      Perform clustered differing of the image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Cluster()
{
    if (!data)
        return false;
    float matrix[4][4] = { {  .75,  .375,  .625,   .25},
                           {.0625,     1,  .875, .4375},
                           {   .5, .8125, .9375, .1250},
                           {.1875, .5625, .3125, .6875} };

    To_Grayscale();

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            int mv = ((i * width) + j) * 4;
            for (int k = 0; k < 3; k++) {
                if (data[mv] < (matrix[i % 4][j % 4] * 255) && data[mv] != 3)
                    data[mv + k] = 0;
                else
                    data[mv + k] = 255;
            }
        }
    return true;
}// Dither_Cluster


///////////////////////////////////////////////////////////////////////////////
//
//  Convert the image to an 8 bit image using Floyd-Steinberg dithering over
//  a uniform quantization - the same quantization as in Quant_Uniform.
//  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Color()
{
    if (!data) {
        return false;
    }
    #define offset(x, y) ((x + (y * width)) * 4)
    float *temp = new float[(width + 2) * (height + 1) * 4];
    float q[3], rgb1[3], rgb2[3];
    const int rgbArray[3] = { 7, 7, 3 };

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int z = 0; z < 3; z++) {
                int tempArg = offset(x + 1, y) + z;
                temp[tempArg] = (float)data[offset(x, y) + z];
            }

    for (int y = 0; y < height; y++)
        for (int x = 1; x <= width; x++)
            for (int i = 0; i < 3; i++) {
                rgb1[i] = temp[offset(x, y) + i];
                rgb2[i] = round(round(rgb1[i] / (255.0 / rgbArray[i])) * (255.0 / rgbArray[i]));
                temp[offset(x, y) + i] = rgb2[i];
                q[i] = rgb1[i] - rgb2[i];
                temp[offset(x - 1, y + 1) + i] = max(0.0, min(temp[offset(x - 1, y + 1) + i] + q[i] * (3.0 / 16), 255.0));
                temp[offset(x, y + 1) + i] = max(0.0, min(temp[offset(x, y + 1) + i] + q[i] * (5.0 / 16), 255.0));
                temp[offset(x + 1, y) + i] = max(0.0, min(temp[offset(x + 1, y) + i] + q[i] * (7.0 / 16), 255.0));
                temp[offset(x + 1, y + 1) + i] = max(0.0, min(temp[offset(x + 1, y + 1) + i] + q[i] * (1.0 / 16), 255.0));                   
        
                rgb1[i] = temp[offset(x, y) + i];
                rgb2[i] = round(round(rgb1[i] / (255.0 / rgbArray[i])) * (255.0 / rgbArray[i]));
                temp[offset(x, y) + i] = rgb2[i];
                q[i] = rgb1[i] - rgb2[i];
                temp[offset(x + 1, y + 1) + i] = temp[offset(x + 1, y + 1) + i] + q[i] * (3.0 / 16);
                temp[offset(x, y + 1) + i] = temp[offset(x, y + 1) + i] + q[i] * (5.0 / 16);
                temp[offset(x - 1, y) + i] = temp[offset(x - 1, y) + i] + q[i] * (7.0 / 16);
                temp[offset(x - 1, y + 1) + i] = temp[offset(x - 1, y + 1) + i] + q[i] * (1.0 / 16);   
            }

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int z = 0; z < 3; z++)
                data[offset(x, y) + z] = temp[offset(x + 1, y) + z];

    delete[] temp;
    return true;
}// Dither_Color


///////////////////////////////////////////////////////////////////////////////
//
//      Composite the current image over the given image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Over(TargaImage* pImage)
{

    if (width != pImage->width || height != pImage->height)
    {
        cout << "Comp_Over: Images not the same size\n";
        return false;
    }
    const int canvasSize = width * height * 4;
    for (int i = 0; i < canvasSize; i += 4) {
        float alpha = ((float)data[i + 3]) / 255.0;

        for (int j = 0; j < 4; j++) {
            if (j < 3)
                data[i + j] *= alpha;
            data[i + j] += (pImage->data[i + j] * (1.0 - alpha));
        }
    }
    return true;
}// Comp_Over


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image "in" the given image.  See lecture notes for 
//  details.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_In(TargaImage* pImage)
{
    if (width != pImage->width || height != pImage->height)
    {
        cout << "Comp_In: Images not the same size\n";
        return false;
    }
    
    const int canvasSize = width * height * 4;
    for (int i = 0; i < canvasSize; i += 4) {
        float alpha = ((float)pImage->data[i + 3]) / 255.0;
        for (int j = 0; j < 4; j++)
            data[i + j] *= alpha;
    }
}// Comp_In


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image "out" the given image.  See lecture notes for 
//  details.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Out(TargaImage* pImage)
{
    if (width != pImage->width || height != pImage->height)
    {
        cout << "Comp_Out: Images not the same size\n";
        return false;
    }
    const int canvasSize = width * height * 4;
    for (int i = 0; i < canvasSize; i += 4) {
        float alpha = ((float)pImage->data[i + 3]) / 255.0;

        for (int j = 0; j < 4; j++)
            data[i + j] *= (1.0 - alpha);
    }
    return true;
}// Comp_Out


///////////////////////////////////////////////////////////////////////////////
//
//      Composite current image "atop" given image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Atop(TargaImage* pImage)
{
    if (width != pImage->width || height != pImage->height)
    {
        cout << "Comp_Atop: Images not the same size\n";
        return false;
    }

    const int canvasSize = width * height * 4;
    for (int i = 0; i < canvasSize; i += 4) {
        float a_f = ((float)data[i + 3]) / 255.0;
        float a_g = ((float)pImage->data[i + 3]) / 255.0;

        for (int j = 0; j < 4; j++) {
            data[i + j] *= a_g;
            data[i + j] += (pImage->data[i + j] * (1.0 - a_f));
        }
    }
    return true;
}// Comp_Atop


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image with given image using exclusive or (XOR).  Return
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Xor(TargaImage* pImage)
{
    if (width != pImage->width || height != pImage->height)
    {
        cout << "Comp_Xor: Images not the same size\n";
        return false;
    }
    const int canvasSize = width * height * 4;
    for (int i = 0; i < canvasSize; i += 4) {
        float a_f = ((float)data[i + 3]) / 255.0;
        float a_g = ((float)pImage->data[i + 3]) / 255.0;

        for (int j = 0; j < 4; j++) {
            data[i + j] *= (1.0 - a_g);
            data[i + j] += (pImage->data[i + j] * (1.0 - a_f));
        }
    }
    return true;
}// Comp_Xor


///////////////////////////////////////////////////////////////////////////////
//
//      Calculate the difference bewteen this imag and the given one.  Image 
//  dimensions must be equal.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Difference(TargaImage* pImage)
{
    if (!pImage)
        return false;

    if (width != pImage->width || height != pImage->height)
    {
        cout << "Difference: Images not the same size\n";
        return false;
    }// if

    for (int i = 0 ; i < width * height * 4 ; i += 4)
    {
        unsigned char        rgb1[3];
        unsigned char        rgb2[3];

        RGBA_To_RGB(data + i, rgb1);
        RGBA_To_RGB(pImage->data + i, rgb2);

        data[i] = abs(rgb1[0] - rgb2[0]);
        data[i+1] = abs(rgb1[1] - rgb2[1]);
        data[i+2] = abs(rgb1[2] - rgb2[2]);
        data[i+3] = 255;
    }

    return true;
}// Difference


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 box filter on this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////

bool TargaImage::Filter_Box()
{
    if (!data) {
        return false;
    }
#define offset(x,y) ((x + (y * width)) * 4)
    const int canvasSize = width * height * 4;

    unsigned char* output = new unsigned char[canvasSize];
    std::memcpy(output, data, width * height * sizeof(data));

    const int edgeLength = 5;

    int* sum = new int[3];

    int count = 0;
    int marginSize = (int)floor(edgeLength / 2.0);
    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++){
            for (int i = max(0, x - marginSize); i <= min(x + marginSize, width - 1); i++) {
                for (int j = 0; j < 3; j++)
                    sum[j] += data[offset(i, y) + j];
                count++;
            }
            for (int j = 0; j < 3; j++) {
                output[offset(x, y) + j] = sum[j] / count;
                sum[j] = 0;
            }
            count = 0;
        }

    std::memcpy(data, output, width * height * sizeof(output));

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++) {
            for (int i = max(0, y - marginSize); i <= min(y + marginSize, height - 1); i++) {
                for (int j = 0; j < 3; j++) {
                    sum[j] += data[offset(x, i) + j];
                }
                count += 1;
            }
            for (int j = 0; j < 3; j++) {
                output[offset(x, y) + j] = sum[j] / count;
                sum[j] = 0;
            }
            count = 0;
        }
    std::memcpy(data, output, width * height * sizeof(output));
    delete[] output;
    return true;
}// Filter_Box



///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 Bartlett filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////

bool TargaImage::Filter_Bartlett()
{
    if (!data) {
        return false;
    }

    const int edgeLength = 5;
    const int canvasSize = width * height * 4;
    #define offset(x, y) ((x + (y * width)) * 4)
    const int filterSize = ceil(edgeLength / 2.0) - 1;
    const int marginSize = (int)floor(edgeLength / 2.0);
    int *filter = new int[edgeLength];
    unsigned char* output = new unsigned char[canvasSize];
    
    for (int i = 0; i < marginSize; i++) {
        filter[i] = i + 1;
        filter[edgeLength - 1 - i] = i + 1;
    }

    filter[filterSize] = filterSize + 1;

    std::memcpy(output, data, width * height * sizeof(data));

    int* sum = new int[3];
    int count = 0;

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++) {
            for (int i = max(0, x - marginSize); i <= min(x + marginSize, width - 1); i++) {
                for (int j = 0; j < 3; j++)
                    sum[j] += data[offset(i, y) + j] * filter[filterSize - abs(i - x)];
                count += filter[filterSize - abs(i - x)];
            }
            for (int j = 0; j < 3; j++) {
                output[offset(x, y) + j] = sum[j] / count;
                sum[j] = 0;
            }
            count = 0;
        }

    std::memcpy(data, output, width * height * sizeof(output));

    for (int x = 0; x < width; x++) 
        for (int y = 0; y < height; y++) {
            for (int j = max(0, y - marginSize); j <= min(y + marginSize, height - 1); j++) {
                for (int i = 0; i < 3; i++)
                    sum[i] += data[offset(x, j) + i] * filter[filterSize - abs(j - y)];
                count += filter[filterSize - abs(j - y)];
            }
            for (int j = 0; j < 3; j++) {
                output[offset(x, y) + j] = sum[j] / count;
                sum[j] = 0;
            }
            count = 0;
        }
    std::memcpy(data, output, width * height * sizeof(output));
    delete[] output, filter;
    return true;
}// Filter_Bartlett


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 Gaussian filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Gaussian()
{
    if (!data) {
        return false;
    }
    const int edgeLength = 5;
    const int canvasSize = width * height * 4;
    #define offset(x, y) ((x + (y * width)) * 4)
    const int filterSize = ceil(edgeLength / 2.0) - 1;
    const int marginSize = (int)floor(edgeLength / 2.0);
    int* filter = new int[edgeLength];

    for (int i = 0; i < (int)floor(edgeLength / 2.0); i++) {
        filter[i] = filter[edgeLength - 1 - i] = Binomial(edgeLength - 1, i);
    }

    filter[filterSize] = Binomial(edgeLength - 1, edgeLength / 2);

    unsigned char* output = new unsigned char[canvasSize];
    std::memcpy(output, data, width * height * sizeof(data));
    
    int* sum = new int[3];
    int count = 0;

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++) {
            for (int i = max(0, x - marginSize); i <= min(x + marginSize, width - 1); i++) {
                for (int j = 0; j < 3; j++)
                    sum[j] += data[offset(i, y) + j] * filter[filterSize - abs(i - x)];
                count += filter[filterSize - abs(i - x)];
            }
            for (int i = 0; i < 3; i++) {
                output[offset(x, y) + i] = sum[i] / count;
                sum[i] = 0;
            }
            count = 0;
        }

    std::memcpy(data, output, width * height * sizeof(output));

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++) {
            for (int j = max(0, y - marginSize); j <= min(y + marginSize, height - 1); j++) {
                for (int i = 0; i < 3; i++)
                    sum[i] += data[offset(x, j) + i] * filter[filterSize - abs(j - y)];
                count += filter[filterSize - abs(j - y)];
            }
            for (int i = 0; i < 3; i++) {
                output[offset(x, y) + i] = sum[i] / count;
                sum[i] = 0;
            }
            count = 0;
        }

    std::memcpy(data, output, width * height * sizeof(output));
    delete[] output, filter;
    return true;
}// Filter_Gaussian

///////////////////////////////////////////////////////////////////////////////
//
//      Perform NxN Gaussian filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////

bool TargaImage::Filter_Gaussian_N( unsigned int N )
{

    if (!data) {
        return false;
    }
    const int edgeLength = N;
    const int canvasSize = width * height * 4;
#define offset(x, y) ((x + (y * width)) * 4)
    const int filterSize = ceil(edgeLength / 2.0) - 1;
    const int marginSize = (int)floor(edgeLength / 2.0);
    int* filter = new int[edgeLength];

    for (int i = 0; i < (int)floor(edgeLength / 2.0); i++) {
        filter[i] = filter[edgeLength - 1 - i] = Binomial(edgeLength - 1, i);
    }

    if (edgeLength % 2 != 0)
        filter[filterSize] = Binomial(edgeLength - 1, edgeLength / 2);

    unsigned char* output = new unsigned char[canvasSize];
    std::memcpy(output, data, width * height * sizeof(data));

    int* sum = new int[3];
    int count = 0;

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++) {
            for (int i = max(0, x - marginSize); i <= min(x + marginSize, width - 1); i++) {
                for (int j = 0; j < 3; j++)
                    sum[j] += data[offset(i, y) + j] * filter[filterSize - abs(i - x)];
                count += filter[filterSize - abs(i - x)];
            }
            for (int i = 0; i < 3; i++) {
                output[offset(x, y) + i] = sum[i] / count;
                sum[i] = 0;
            }
            count = 0;
        }

    std::memcpy(data, output, width * height * sizeof(output));

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++) {
            for (int j = max(0, y - marginSize); j <= min(y + marginSize, height - 1); j++) {
                for (int i = 0; i < 3; i++)
                    sum[i] += data[offset(x, j) + i] * filter[filterSize - abs(j - y)];
                count += filter[filterSize - abs(j - y)];
            }
            for (int i = 0; i < 3; i++) {
                output[offset(x, y) + i] = sum[i] / count;
                sum[i] = 0;
            }
            count = 0;
        }

    std::memcpy(data, output, width * height * sizeof(output));
    delete[] output, filter;
    return true;
}// Filter_Gaussian_N


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 edge detect (high pass) filter on this image.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Edge()
{
    ClearToBlack();
    return false;
}// Filter_Edge


///////////////////////////////////////////////////////////////////////////////
//
//      Perform a 5x5 enhancement filter to this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Enhance()
{
    ClearToBlack();
    return false;
}// Filter_Enhance


///////////////////////////////////////////////////////////////////////////////
//
//      Run simplified version of Hertzmann's painterly image filter.
//      You probably will want to use the Draw_Stroke funciton and the
//      Stroke class to help.
// Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::NPR_Paint()
{
    ClearToBlack();
    return false;
}



///////////////////////////////////////////////////////////////////////////////
//
//      Halve the dimensions of this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Half_Size()
{
    ClearToBlack();
    return false;
}// Half_Size


///////////////////////////////////////////////////////////////////////////////
//
//      Double the dimensions of this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Double_Size()
{
    ClearToBlack();
    return false;
}// Double_Size


///////////////////////////////////////////////////////////////////////////////
//
//      Scale the image dimensions by the given factor.  The given factor is 
//  assumed to be greater than one.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Resize(float scale)
{
    ClearToBlack();
    return false;
}// Resize


//////////////////////////////////////////////////////////////////////////////
//
//      Rotate the image clockwise by the given angle.  Do not resize the 
//  image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Rotate(float angleDegrees)
{
    ClearToBlack();
    return false;
}// Rotate


//////////////////////////////////////////////////////////////////////////////
//
//      Given a single RGBA pixel return, via the second argument, the RGB
//      equivalent composited with a black background.
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::RGBA_To_RGB(unsigned char *rgba, unsigned char *rgb)
{
    const unsigned char	BACKGROUND[3] = { 0, 0, 0 };

    unsigned char  alpha = rgba[3];

    if (alpha == 0)
    {
        rgb[0] = BACKGROUND[0];
        rgb[1] = BACKGROUND[1];
        rgb[2] = BACKGROUND[2];
    }
    else
    {
	    float	alpha_scale = (float)255 / (float)alpha;
	    int	val;
	    int	i;

	    for (i = 0 ; i < 3 ; i++)
	    {
	        val = (int)floor(rgba[i] * alpha_scale);
	        if (val < 0)
		    rgb[i] = 0;
	        else if (val > 255)
		    rgb[i] = 255;
	        else
		    rgb[i] = val;
	    }
    }
}// RGA_To_RGB


///////////////////////////////////////////////////////////////////////////////
//
//      Copy this into a new image, reversing the rows as it goes. A pointer
//  to the new image is returned.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage* TargaImage::Reverse_Rows(void)
{
    unsigned char   *dest = new unsigned char[width * height * 4];
    TargaImage	    *result;
    int 	        i, j;

    if (! data)
    	return NULL;

    for (i = 0 ; i < height ; i++)
    {
	    int in_offset = (height - i - 1) * width * 4;
	    int out_offset = i * width * 4;

	    for (j = 0 ; j < width ; j++)
        {
	        dest[out_offset + j * 4] = data[in_offset + j * 4];
	        dest[out_offset + j * 4 + 1] = data[in_offset + j * 4 + 1];
	        dest[out_offset + j * 4 + 2] = data[in_offset + j * 4 + 2];
	        dest[out_offset + j * 4 + 3] = data[in_offset + j * 4 + 3];
        }
    }

    result = new TargaImage(width, height, dest);
    delete[] dest;
    return result;
}// Reverse_Rows


///////////////////////////////////////////////////////////////////////////////
//
//      Clear the image to all black.
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::ClearToBlack()
{
    memset(data, 0, width * height * 4);
}// ClearToBlack


///////////////////////////////////////////////////////////////////////////////
//
//      Helper function for the painterly filter; paint a stroke at
// the given location
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::Paint_Stroke(const Stroke& s) {
   int radius_squared = (int)s.radius * (int)s.radius;
   for (int x_off = -((int)s.radius); x_off <= (int)s.radius; x_off++) {
      for (int y_off = -((int)s.radius); y_off <= (int)s.radius; y_off++) {
         int x_loc = (int)s.x + x_off;
         int y_loc = (int)s.y + y_off;
         // are we inside the circle, and inside the image?
         if ((x_loc >= 0 && x_loc < width && y_loc >= 0 && y_loc < height)) {
            int dist_squared = x_off * x_off + y_off * y_off;
            if (dist_squared <= radius_squared) {
               data[(y_loc * width + x_loc) * 4 + 0] = s.r;
               data[(y_loc * width + x_loc) * 4 + 1] = s.g;
               data[(y_loc * width + x_loc) * 4 + 2] = s.b;
               data[(y_loc * width + x_loc) * 4 + 3] = s.a;
            } else if (dist_squared == radius_squared + 1) {
               data[(y_loc * width + x_loc) * 4 + 0] = 
                  (data[(y_loc * width + x_loc) * 4 + 0] + s.r) / 2;
               data[(y_loc * width + x_loc) * 4 + 1] = 
                  (data[(y_loc * width + x_loc) * 4 + 1] + s.g) / 2;
               data[(y_loc * width + x_loc) * 4 + 2] = 
                  (data[(y_loc * width + x_loc) * 4 + 2] + s.b) / 2;
               data[(y_loc * width + x_loc) * 4 + 3] = 
                  (data[(y_loc * width + x_loc) * 4 + 3] + s.a) / 2;
            }
         }
      }
   }
}


///////////////////////////////////////////////////////////////////////////////
//
//      Build a Stroke
//
///////////////////////////////////////////////////////////////////////////////
Stroke::Stroke() {}

///////////////////////////////////////////////////////////////////////////////
//
//      Build a Stroke
//
///////////////////////////////////////////////////////////////////////////////
Stroke::Stroke(unsigned int iradius, unsigned int ix, unsigned int iy,
               unsigned char ir, unsigned char ig, unsigned char ib, unsigned char ia) :
   radius(iradius),x(ix),y(iy),r(ir),g(ig),b(ib),a(ia)
{
}

