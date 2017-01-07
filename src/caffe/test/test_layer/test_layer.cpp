#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
#include <assert.h>
using namespace std;

#include "caffe/blob.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/get.hpp"
#include "caffe/filler.hpp"
using namespace caffe;

#define Dtype float16
#define Mtype float16

class LayerTest {
public:
    LayerTest() {
        Caffe::set_mode(Caffe::GPU);
        init();
    }

    void ConvolutionLayerTest() {
        // Fill bottom blob
        init_rand(bottom_blob); 
        cout << "I: " << to_string(bottom_blob->shape()) << endl;
        clog << to_string(bottom_blob) << endl;

        // Set up layer parameters
        LayerParameter layer_param;
        ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
        conv_param->add_kernel_size(3);
        conv_param->add_stride(2);
        conv_param->set_num_output(4); // number of filters
        //conv_param->mutable_weight_filler()->set_type("constant"); // type of filters
        //conv_param->mutable_weight_filler()->set_value(1.0);
        //conv_param->mutable_weight_filler()->set_type("gaussian");

        // Create layer
        std::shared_ptr<Layer<Dtype,Mtype> > layer(new ConvolutionLayer<Dtype,Mtype>(layer_param));
        layer->SetUp(bottom,top);

        assert(top_blob->num() == num);
        assert(top_blob->channels() == 4);
        assert(top_blob->height() == 1);
        assert(top_blob->width() == 2);

        Blob<Dtype,Mtype>* weights = layer->blobs()[0].get();
        init_rand(weights);
        cout << "W: " << to_string(weights->shape()) << endl;
        clog << to_string(weights) << endl;

        // Run forward pass
        timespec start,end,elapsed;
        clock_gettime(CLOCK_REALTIME,&start);
        layer->Forward(bottom,top);
        clock_gettime(CLOCK_REALTIME,&end);
        cout << "O: " << to_string(top_blob->shape()) << endl;
        clog << to_string(top_blob) << endl;
        elapsed = diff(start,end);
        cout<<"time: "<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;
    }

    void InnerProductLayerTest() {
        // Fill bottom blob
        init_rand(bottom_blob);
        cout << "I: " << to_string(bottom_blob->shape()) << endl; 
        clog << to_string(bottom_blob) << endl;

        // Set up layer parameters
        LayerParameter layer_param;
        InnerProductParameter* inner_product_param = layer_param.mutable_inner_product_param();
        inner_product_param->set_num_output(10);
        //inner_product_param->mutable_weight_filler()->set_type("constant");
        //inner_product_param->mutable_weight_filler()->set_value(1.0);
        //inner_product_param->mutable_weight_filler()->set_type("uniform");

        // Create layer
        std::shared_ptr<Layer<Dtype,Mtype> > layer(new InnerProductLayer<Dtype,Mtype>(layer_param));
        layer->SetUp(bottom,top);

        assert(top_blob->num() == num);
        assert(top_blob->channels() == 10);
        assert(top_blob->height() == 1);
        assert(top_blob->width() == 1);

        Blob<Dtype,Mtype>* weights = layer->blobs()[0].get();
        init_rand(weights);
        cout << "W: " << to_string(weights->shape()) << endl;
        clog << to_string(weights) << endl;

        // Run forward pass
        timespec start,end,elapsed;
        clock_gettime(CLOCK_REALTIME,&start);
        layer->Forward(bottom,top);
        clock_gettime(CLOCK_REALTIME,&end);
        cout << "O: " << to_string(top_blob->shape()) << endl;
        clog << to_string(top_blob) << endl;
        elapsed = diff(start,end);
        cout<<"time: "<<elapsed.tv_sec*1000000000+elapsed.tv_nsec<<endl;
    }

private:
    int num = 2;
    int channels = 3;
    int height = 4;
    int width = 5;

    Blob<Dtype,Mtype>* bottom_blob;
    Blob<Dtype,Mtype>* top_blob;
    vector<Blob<Dtype,Mtype>*> bottom;
    vector<Blob<Dtype,Mtype>*> top;

    ofstream ofs;

    void init() {
        // Create bottom blob
        bottom_blob = new Blob<Dtype,Mtype>();
        bottom.push_back(bottom_blob);

        // Reshape bottom blob
        vector<int> shape {num, channels, height, width};
        bottom_blob->Reshape(shape);
        
        // Create top blob
        top_blob = new Blob<Dtype,Mtype>();
        top.push_back(top_blob);

        ofs.open("test_layer.log");
        clog.rdbuf(ofs.rdbuf());
    }

    string to_string(vector<int> shape) {
        string s = "";
        s += "(" + std::to_string(shape[0]);
        for (int i = 1; i < shape.size(); ++i) {
            s += "," + std::to_string(shape[i]);
        }
        s += ")";
        return s;
    }

    string to_string(Blob<Dtype,Mtype>* blob) {
        const Dtype* data = blob->cpu_data();
        int n = blob->num();
        int c = blob->channels();
        int h = blob->height();
        int w = blob->width();
        string s = "";
        for (int i = 0; i < n*c*h*w; ++i) {
            s += std::to_string(data[i]) + " ";
        }
        return s;
    }

    void init_rand(Blob<Dtype,Mtype>* blob) {
        Dtype* data = blob->mutable_cpu_data();
        for (int i = 0; i < blob->count(); ++i) {
            data[i] = Get<float16>(float(rand())/float(RAND_MAX));
        }
    }

    void init_ones(Blob<Dtype,Mtype>* blob) {
        Dtype* data = blob->mutable_cpu_data();
        srand(1234);
        for (int i = 0; i < blob->count(); ++i) {
            data[i] = Get<float16>(1.);
        }
    }

    timespec diff(timespec start, timespec end) {
        timespec temp;
        if ((end.tv_nsec-start.tv_nsec)<0) {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
        } 
        else {
            temp.tv_sec = end.tv_sec-start.tv_sec;
            temp.tv_nsec = end.tv_nsec-start.tv_nsec;
       }
       return temp;
    }
};


int main() {
    LayerTest test;
    //test.ConvolutionLayerTest();
    test.InnerProductLayerTest();
}
