#include <vector>
#include <string>
using namespace std;

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/get.hpp"
using namespace caffe;

void print_shape(Blob<float16,float16>* blob) {
    vector<int> shape = blob->shape();
    cout << "(" << shape[0];
    for (int i = 1; i < shape.size(); ++i) {
        cout << "," << shape[i];
    }
    cout << ")" << endl;
}

int main() {
    vector<Blob<float16,float16>*> bottom;
    Blob<float16,float16>* bottom_blob = new Blob<float16,float16>(2, 3, 7, 5);
    bottom.push_back(bottom_blob);

    vector<Blob<float16,float16>*> top;
    Blob<float16,float16>* top_blob = new Blob<float16,float16>();
    top.push_back(top_blob);

    LayerParameter layer_param;
    ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
    conv_param->add_kernel_size(3);
    conv_param->add_stride(2);
    conv_param->set_num_output(4); // number of filters
    
    shared_ptr<Layer<float16,float16> > layer(new ConvolutionLayer<float16,float16>(layer_param));
    layer->SetUp(bottom,top);
    
    print_shape(top_blob);
    //EXPECT_EQ(top_blob->num(), 2);
    //EXPECT_EQ(top_blob->channels(), 4);
    //EXPECT_EQ(top_blob->height(), 3);
    //EXPECT_EQ(top_blob->width(), 2);
    
    layer->Forward(bottom,top);
    const float16* top_data = top_blob->cpu_data();
    for (int i = 0; i < top_blob->count(); ++i)
        cout << top_data[i] << " ";
    cout << endl;
}
