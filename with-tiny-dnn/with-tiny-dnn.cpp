
#include "tiny_dnn/tiny_dnn.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <iostream>

#include <png.h>

#include "../weights.hpp"

//#define VERBOSE 1

// Much of the code in this file just addresses the question: given
// our nested vectors of weights indexed in HWCK order, how do we dump
// them into the flat weight arrays used by tiny-dnn in KCHW order?
// And how do we ensure that the dense layers, which were trained
// using flattened input from HWC ordered convolutions, work with this
// layout? This would be simpler if we had CHW indexing (Keras
// "theano" or "channels_first" order), and far simpler if we had
// trained and saved the model using tiny-dnn in the first place.

using namespace tiny_dnn;
using std::vector;
using std::map;
using std::cout;
using std::cerr;
using std::endl;
using std::string;

vector<float>
read_image(string filename)
{
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;

    if (!png_image_begin_read_from_file(&image, filename.c_str())) {
        throw std::runtime_error
            (string("Failed to open image file: ") + image.message);
    }

    image.format = PNG_FORMAT_RGB;
    auto size = image.width * image.height * 3;
    vector<png_byte> buffer(size, 255);
    
    if (!png_image_finish_read(&image, nullptr, buffer.data(), 0, nullptr)) {
        throw std::runtime_error
            (string("Failed to read image file: ") + image.message);
    }

    vector<float> scaled;
    scaled.reserve(size);

    for (png_uint_32 k = 0; k < 3; ++k) {
        for (png_uint_32 i = 0; i < image.height; ++i) {
            for (png_uint_32 j = 0; j < image.width; ++j) {
                scaled.push_back(buffer[i * (image.width * 3) +
                                        j * 3 +
                                        k]
                                 / 255.f);
            }
        }
    }
    
    return scaled;
}

void
load_convolution_weights(convolutional_layer &layer,
                         const vector<vector<vector<vector<float>>>> &weights,
                         const vector<float> &biases)
{
    size_t kernel_height = weights.size();
    size_t kernel_width = weights[0].size();
    size_t input_depth = weights[0][0].size();
    size_t nkernels = weights[0][0][0].size();

    auto weight_ptr = layer.weights()[0]->begin();

#ifdef VERBOSE
    cerr << "expected weights = " << kernel_height * kernel_width * nkernels * input_depth << ", weights = " << layer.weights()[0]->size() << ", expected biases = " << nkernels << ", biases = " << layer.weights()[1]->size() << endl;
#endif
    
    for (size_t k = 0; k < nkernels; ++k) {
        for (size_t c = 0; c < input_depth; ++c) {
            for (size_t y = 0; y < kernel_height; ++y) {
                for (size_t x = 0; x < kernel_width; ++x) {
                    *weight_ptr++ = weights[y][x][c][k];
                }
            }
        }
    }
    
    auto bias_ptr = layer.weights()[1]->begin();

    for (size_t k = 0; k < nkernels; ++k) {
        *bias_ptr++ = biases[k];
    }
}

void
reorder_dense_weights(fully_connected_layer &layer,
                      size_t height,
                      size_t width,
                      size_t depth,
                      size_t out_size)
{
    auto &weights = *layer.weights()[0];
    auto original_weights = weights;

    // Rearrange the weights for a fully-connected layer, so that a
    // layer that has been trained using HWC (channels-last) layout
    // can be fed data in CHW (channels-first) form. We could do this
    // while loading them in the first place of course
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            for (size_t c = 0; c < depth; ++c) {
                size_t in_ix = y * width * depth + x * depth + c;
                size_t out_ix = c * height * width + y * width + x;
                for (size_t i = 0; i < out_size; ++i) {
                    weights[out_ix * out_size + i] =
                        original_weights[in_ix * out_size + i];
                }
            }
        }
    }
}

void
load_dense_weights(fully_connected_layer &layer,
                   const vector<vector<float>> &weights,
                   const vector<float> &biases)
{
    size_t in_size = weights.size();
    size_t out_size = weights[0].size();
    
    auto weight_ptr = layer.weights()[0]->begin();

#ifdef VERBOSE
    cerr << "in_size = " << in_size << ", out_size = " << out_size << ", weights = " << layer.weights()[0]->size() << ", biases = " << layer.weights()[1]->size() << endl;
#endif

    for (size_t i = 0; i < in_size; ++i) {
        for (size_t j = 0; j < out_size; ++j) {
            *weight_ptr++ = weights[i][j];
        }
    }

    auto bias_ptr = layer.weights()[1]->begin();

    for (size_t j = 0; j < out_size; ++j) {
        *bias_ptr++ = biases[j];
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <filename.png>" << endl;
        return 2;
    }

    vector<float> image = read_image(argv[1]);

    size_t expected_width = 128;
    size_t expected_height = 128;
    if (image.size() != expected_height * expected_width * 3) {
        cerr << "Image has wrong size: must be exactly " << expected_width
             << "x" << expected_height << " pixels" << endl;
        throw std::runtime_error("Image has wrong size");
    }

    network<sequential> model;

    convolutional_layer first_conv { 128, 128, 3, 3, 32, padding::same, true };
    convolutional_layer second_conv { 64, 64, 3, 32, 16, padding::same, true };
    convolutional_layer third_conv { 32, 32, 3, 16, 16, padding::same, true };
    convolutional_layer fourth_conv { 16, 16, 3, 16, 8, padding::same, true };
    fully_connected_layer first_dense { 512, 256, true };
    fully_connected_layer labeller { 256, 5, true };
    
    model << first_conv
          << relu_layer(128, 128, 32)
          << max_pooling_layer(128, 128, 32, 2)
          << second_conv
          << relu_layer(64, 64, 16)
          << max_pooling_layer(64, 64, 16, 2)
          << third_conv
          << relu_layer(32, 32, 16)
          << max_pooling_layer(32, 32, 16, 2)
          << fourth_conv
          << relu_layer(16, 16, 8)
          << max_pooling_layer(16, 16, 8, 2)
          << first_dense
          << relu_layer(256)
          << labeller
          << softmax_layer(5);

    first_conv.init_weight();
    second_conv.init_weight();
    third_conv.init_weight();
    fourth_conv.init_weight();
    first_dense.init_weight();
    labeller.init_weight();

    load_convolution_weights(first_conv, weights_firstConv, biases_firstConv);
    load_convolution_weights(second_conv, weights_secondConv, biases_secondConv);
    load_convolution_weights(third_conv, weights_thirdConv, biases_thirdConv);
    load_convolution_weights(fourth_conv, weights_fourthConv, biases_fourthConv);
    load_dense_weights(first_dense, weights_firstDense, biases_firstDense);
    reorder_dense_weights(first_dense, 8, 8, 8, 256);
    load_dense_weights(labeller, weights_labeller, biases_labeller);
/*
    model.save("model.json",
               content_type::weights_and_model,
               file_format::json);
*/    
    auto result = model.predict(image);
    
    vector<string> labels {
        "daisy", "dandelion", "roses", "sunflowers", "tulips"
    };

    map<float, string> ranking;
    
    for (size_t i = 0; i < result.size(); ++i) {
        if (i >= labels.size()) {
            throw std::logic_error("Too many result categories!");
        }
        ranking[- result[i] * 100.0] = labels[i];
    }

    for (auto r = ranking.begin(); r != ranking.end(); ++r) {
        cout << r->second << ": " << -r->first << "%" << endl;
    }
    
    return 0;
}

