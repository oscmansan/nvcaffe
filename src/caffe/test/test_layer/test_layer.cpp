#include <vector>
#include <string>
using namespace std;

#include "caffe/blob.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/get.hpp"
using namespace caffe;


int main() {
    string param_file = "conv_layer.prototxt";
    LayerParameter param;
    if (!ReadProtoFromTextFile(param_file, &param))
        cout << "Failed to parse LayerParameter file: " << param_file << endl;
    
    ConvolutionLayer<float16,float16> layer(param);
    ConvolutionParameter conv_param = layer.layer_param().convolution_param();

    vector<Blob<float16,float16>*> bottom;
    Blob<float16,float16>* bottom_blob = new Blob<float16,float16>(1,3,600,1000);
    bottom.push_back(bottom_blob);
    cout << "num_axes: " << bottom_blob->num_axes() << endl;
    const int channel_axis_ = bottom_blob->CanonicalAxisIndex(conv_param.axis());
    const int first_spatial_axis = channel_axis_ + 1;
    const int num_axes = bottom_blob->num_axes();
    const int num_spatial_axes_ = num_axes - first_spatial_axis;
    cout << "firs_spatial_axis: " << first_spatial_axis << endl;
    cout << "num_spatial_axes: " << num_spatial_axes_ << endl;

    vector<Blob<float16,float16>*> top;
    Blob<float16,float16>* top_blob = new Blob<float16,float16>(1,64,600,1000);
    top.push_back(top_blob);

    layer.LayerSetUp(bottom, top);
    layer.Forward(bottom, top);
    
    vector<int> shape = top[0]->shape();
    cout << "(" << shape[0];
    for (int i = 1; i < shape.size(); ++i) {
        cout << "," << shape[i];
    }
    cout << ")" << endl;
}
