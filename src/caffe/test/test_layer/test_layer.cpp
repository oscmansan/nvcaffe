#include <vector>
#include <string>
using namespace std;

#include "gtest/gtest.h"
#include <assert.h>

#include "caffe/blob.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/get.hpp"
#include "caffe/filler.hpp"
using namespace caffe;

void print_shape(Blob<float16,float16>* blob) {
    vector<int> shape = blob->shape();
    cout << "(" << shape[0];
    for (int i = 1; i < shape.size(); ++i) {
        cout << "," << shape[i];
    }
    cout << ")" << endl;
}

void print_blob(Blob<float16,float16>* blob) {
    const float16* data = blob->cpu_data();
    for (int i = 0; i < blob->count(); ++i)
        cout << data[i] << " ";
    cout << endl;
}

void init_blob(Blob<float16,float16>* blob) {
    float16* data = blob->mutable_cpu_data();
    for (int i = 0; i < blob->count(); ++i) {
        //data[i] = static_cast<float16>(rand()) / static_cast<float16>(RAND_MAX/1000.);
        data[i] = float16(rand() % 1000);
    }
}


int main() {
    vector<Blob<float16,float16>*> bottom;
    Blob<float16,float16>* bottom_blob = new Blob<float16,float16>(2, 3, 7, 5);
    FillerParameter filler_param;
    filler_param.set_value(1.);
    //GaussianFiller<float16,float16> filler(filler_param);
    //filler.Fill(bottom_blob); print_blob(bottom_blob);
    init_blob(bottom_blob); print_blob(bottom_blob);
    bottom.push_back(bottom_blob);

    vector<Blob<float16,float16>*> top;
    Blob<float16,float16>* top_blob = new Blob<float16,float16>();
    top.push_back(top_blob);

    LayerParameter layer_param;
    ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
    conv_param->add_kernel_size(3);
    conv_param->add_stride(2);
    conv_param->set_num_output(4); // number of filters
    conv_param->mutable_weight_filler()->set_type("gaussian");
    
    shared_ptr<Layer<float16,float16> > layer(new ConvolutionLayer<float16,float16>(layer_param));
    layer->SetUp(bottom,top);
    
    print_shape(top_blob);
    //EXPECT_EQ(top_blob->num(), 2);
    assert(top_blob->num() == 2);
    //EXPECT_EQ(top_blob->channels(), 4);
    assert(top_blob->channels() == 4);
    //EXPECT_EQ(top_blob->height(), 3);
    assert(top_blob->height() == 3);
    //EXPECT_EQ(top_blob->width(), 2);
    assert(top_blob->width() == 2);
    
    layer->Forward(bottom,top);

    print_blob(top_blob);
}
