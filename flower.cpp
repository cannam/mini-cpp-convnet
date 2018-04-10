
#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <iostream>

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <png.h>

#include "weights.hpp"

//#define VERBOSE 1

using namespace std;

vector<vector<vector<float>>>
convolve(const vector<vector<vector<float>>> &in,
         const vector<vector<vector<vector<float>>>> &weights,
         const vector<float> &biases)
{
    // Weights tensor is indexed HWCK, i.e. height - width - number of
    // input channels (e.g. image depth) - number of output channels
    // (i.e. number of learned kernels)
    
    size_t kernel_height = weights.size();
    size_t kernel_width = weights[0].size();
    size_t nkernels = weights[0][0][0].size();
    
    size_t out_height = in.size();
    if (out_height < kernel_height - 1) {
        throw runtime_error("Input too small in convolve");
    }

    size_t out_width = in[0].size();
    if (out_width < kernel_width - 1) {
        throw runtime_error("Input too small in convolve");
    }

    size_t depth = in[0][0].size();
    if (depth != weights[0][0].size()) {
        cerr << "convolve: input depth is " << depth << " but we expected "
             << weights[0][0].size() << endl;
        throw runtime_error("Depth mismatch in convolve");
    }
        
    out_height -= kernel_height - 1;
    out_width -= kernel_width - 1;

#ifdef VERBOSE
    cerr << "convolve: " << nkernels << " kernels of size "
         << kernel_width << "x" << kernel_height << "; output size "
         << out_width << "x" << out_height << "; input depth " << depth << endl;
#endif

    auto out =
        vector<vector<vector<float>>>
        (out_height,
         vector<vector<float>>
         (out_width,
          vector<float>
          (nkernels,
           0.f)));

    for (size_t k = 0; k < nkernels; ++k) {
        for (size_t y = 0; y < out_height; ++y) {
            for (size_t x = 0; x < out_width; ++x) {
                for (size_t c = 0; c < depth; ++c) {
                    for (size_t ky = 0; ky < kernel_height; ++ky) {
                        for (size_t kx = 0; kx < kernel_width; ++kx) {
                            out[y][x][k] +=
                                weights[ky][kx][c][k] * in[y + ky][x + kx][c];
                        }
                    }
                }
                out[y][x][k] += biases[k];
            }
        }
    }

    return out;
}

vector<vector<vector<float>>>
convolutionActivation(const vector<vector<vector<float>>> &in,
                      string type)
{
    auto out(in);

    if (type == "relu") {
        for (size_t i = 0; i < out.size(); ++i) {
            for (size_t j = 0; j < out[i].size(); ++j) {
                for (size_t k = 0; k < out[i][j].size(); ++k) {
                    if (out[i][j][k] < 0.f) {
                        out[i][j][k] = 0.f;
                    }
                }
            }
        }
    } else {
        throw runtime_error("Unknown activation function '" + type + "'");
    }
        
    return out;
}

vector<vector<vector<float>>>
maxPool(const vector<vector<vector<float>>> &in,
        size_t pool_y,
        size_t pool_x)
{
    size_t out_height = in.size() / pool_y;
    if (out_height < 1) {
        throw runtime_error("Input too small in maxPool");
    }

    size_t out_width = in[0].size() / pool_x;
    if (out_width < 1) {
        throw runtime_error("Input too small in maxPool");
    }
    
    size_t depth = in[0][0].size();
    
#ifdef VERBOSE
    cerr << "maxPool: input size " << in[0].size() << "x" << in.size()
         << "; pool size " << pool_x << "x" << pool_y << "; output size "
         << out_width << "x" << out_height << " and depth " << depth << endl;
#endif
    
    auto out =
        vector<vector<vector<float>>>
        (out_height,
         vector<vector<float>>
         (out_width,
          vector<float>
          (depth,
           -INFINITY)));

    for (size_t y = 0; y < out_height; ++y) {
        for (size_t x = 0; x < out_width; ++x) {
            for (size_t i = 0; i < pool_y; ++i) {
                for (size_t j = 0; j < pool_x; ++j) {
                    for (size_t c = 0; c < depth; ++c) {
                        float value = in[y * pool_y + i][x * pool_x + j][c];
                        out[y][x][c] = max(out[y][x][c], value);
                    }
                }
            }
        }
    }

    return out;
}

vector<vector<vector<float>>>
zeroPad(const vector<vector<vector<float>>> &in,
        size_t pad_y,
        size_t pad_x)
{
    size_t in_height = in.size();
    if (in_height == 0) {
        throw runtime_error("Input too small in zeroPad");
    }

    size_t in_width = in[0].size();
    if (in_width == 0) {
        throw runtime_error("Input too small in zeroPad");
    }
    
    size_t depth = in[0][0].size();
    
#ifdef VERBOSE
    cerr << "zeroPad: input size " << in_width << "x" << in_height
         << "; padding " << pad_x << "," << pad_y << "; output size "
         << in_width + 2 * pad_x << "x" << in_height + 2 * pad_y
         << " and depth " << depth << endl;
#endif
    
    auto out =
        vector<vector<vector<float>>>
        (in_height + 2 * pad_y,
         vector<vector<float>>
         (in_width + 2 * pad_x,
          vector<float>
          (depth,
           0.f)));

    for (size_t y = 0; y < in_height; ++y) {
        for (size_t x = 0; x < in_width; ++x) {
            for (size_t c = 0; c < depth; ++c) {
                out[y + pad_y][x + pad_x][c] = in[y][x][c];
            }
        }
    }

    return out;
}

vector<float>
flatten(const vector<vector<vector<float>>> &in)
{
    size_t height = in.size();
    if (height < 1) {
        throw runtime_error("Input too small in flatten");
    }

    size_t width = in[0].size();
    if (width < 1) {
        throw runtime_error("Input too small in flatten");
    }
    
    size_t depth = in[0][0].size();

#ifdef VERBOSE
    cerr << "flatten: input size " << in[0].size() << "x" << in.size()
         << " and depth " << depth << ", output length "
         << width * height * depth << endl;
#endif
    
    vector<float> out(width * height * depth, 0.f);

    size_t i = 0;
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            for (size_t c = 0; c < depth; ++c) {
                out[i++] = in[y][x][c];
            }
        }
    }

    return out;
}

vector<float>
dense(const vector<float> &in,
      const vector<vector<float>> &weights,
      const vector<float> &biases)
{
    size_t in_size = in.size();
    if (in_size != weights.size() || in_size == 0) {
        cerr << "dense: in_size = " << in_size
             << " but we expected " << weights.size() << endl;
        throw runtime_error("Input size mismatch in dense");
    }

    size_t out_size = weights[0].size();
    if (out_size != biases.size() || out_size == 0) {
        cerr << "dense: out_size = " << out_size
             << " but we expected " << biases.size() << endl;
        throw runtime_error("Output size mismatch in dense");
    }
    
#ifdef VERBOSE
    cerr << "dense: input length " << in_size << ", output length "
         << out_size << endl;
#endif

    vector<float> out(out_size, 0.f);

    for (size_t i = 0; i < in_size; ++i) {
        for (size_t j = 0; j < out_size; ++j) {
            out[j] += weights[i][j] * in[i];
        }
    }

    for (size_t j = 0; j < out_size; ++j) {
        out[j] += biases[j];
    }

    return out;
}

vector<float>
denseActivation(const vector<float> &in,
                string type)
{
    auto out(in);
    size_t sz = out.size();

    if (type == "relu") {
        for (size_t i = 0; i < sz; ++i) {
            if (out[i] < 0.f) {
                out[i] = 0.f;
            }
        }
    } else if (type == "softmax") {
        float sum = 0.f;
        for (size_t i = 0; i < sz; ++i) {
            out[i] = exp(out[i]);
            sum += out[i];
        }
        for (size_t i = 0; i < sz; ++i) {
            out[i] /= sum;
        }
    } else {
        throw runtime_error("Unknown activation function '" + type + "'");
    }

    return out;
}

vector<float>
classify(const vector<vector<vector<float>>> &image)
{
    vector<vector<vector<float>>> t;

    t = zeroPad(image, 1, 1);
    t = convolve(t, weights_firstConv, biases_firstConv);
    t = convolutionActivation(t, "relu");
    t = maxPool(t, 2, 2);

    t = zeroPad(t, 1, 1);
    t = convolve(t, weights_secondConv, biases_secondConv);
    t = convolutionActivation(t, "relu");
    t = maxPool(t, 2, 2);

    t = zeroPad(t, 1, 1);
    t = convolve(t, weights_thirdConv, biases_thirdConv);
    t = convolutionActivation(t, "relu");
    t = maxPool(t, 2, 2);

    t = zeroPad(t, 1, 1);
    t = convolve(t, weights_fourthConv, biases_fourthConv);
    t = convolutionActivation(t, "relu");
    t = maxPool(t, 2, 2);

    vector<float> flat = flatten(t);
    
    flat = dense(flat, weights_firstDense, biases_firstDense);
    flat = denseActivation(flat, "relu");

    flat = dense(flat, weights_labeller, biases_labeller);
    flat = denseActivation(flat, "softmax");

    return flat;
}

// Read an image from a PNG file, using libpng, into a tensor in
// format HWC (height-width-channels)

vector<vector<vector<float>>>
read_image(string filename)
{
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;

    if (!png_image_begin_read_from_file(&image, filename.c_str())) {
        throw runtime_error
            (string("Failed to open image file: ") + image.message);
    }

    image.format = PNG_FORMAT_RGB;
    auto size = image.width * image.height * 3;
    vector<png_byte> buffer(size, 255);
    
    if (!png_image_finish_read(&image, nullptr, buffer.data(), 0, nullptr)) {
        throw runtime_error
            (string("Failed to read image file: ") + image.message);
    }

    vector<vector<vector<float>>> image_tensor;
    auto ptr = buffer.begin();
    
    for (png_uint_32 i = 0; i < image.height; ++i) {
        image_tensor.push_back(vector<vector<float>>());
        for (png_uint_32 j = 0; j < image.width; ++j) {
            image_tensor[i].push_back(vector<float>());
            for (png_uint_32 k = 0; k < 3; ++k) {
                image_tensor[i][j].push_back(*ptr++ / 255.f);
            }
        }
    }
        
    return image_tensor;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <filename.png>" << endl;
        return 2;
    }

    auto image = read_image(argv[1]);

    size_t expected_width = 128;
    size_t expected_height = 128;

    if (image.size() != expected_height ||
        image[0].size() != expected_width) {
        cerr << "Image has wrong size: must be exactly " << expected_width
             << "x" << expected_height << " pixels" << endl;
        throw runtime_error("Image has wrong size");
    }

    auto result = classify(image);

    vector<string> labels {
        "daisy", "dandelion", "roses", "sunflowers", "tulips"
    };

    map<float, string> ranking;
    
    for (size_t i = 0; i < result.size(); ++i) {
        if (i >= labels.size()) {
            throw logic_error("Too many result categories!");
        }
        ranking[- result[i] * 100.0] = labels[i];
    }

    for (auto r = ranking.begin(); r != ranking.end(); ++r) {
        cout << r->second << ": " << -r->first << "%" << endl;
    }
    
    return 0;
}

