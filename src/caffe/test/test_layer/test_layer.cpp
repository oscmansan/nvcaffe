#include <vector>

#include "caffe/vision_layers.hpp"
#include "caffe/util/upgrade_proto.hpp"

#define Dtype float16
#define Mtype float16

int main() {
    string param_file = "conv_layer.prototxt";
    LayerParameter param;
    ReadNetParamsFromTextFileOrDie(param_file, &param);
    ConvolutionLayer<Dtype,Mtype> layer(param);

    vector<Blob<Dtype,Mtype>*> bottom;
    vector<Blob<Dtype,Mtype>*> top;
    layer.Forward_cpu(bottom, top);
}
