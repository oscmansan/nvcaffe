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
    cout << layer.type() << endl;

    vector<Blob<float16,float16>*> bottom;
    vector<Blob<float16,float16>*> top;
    layer.Forward(bottom, top);
}
