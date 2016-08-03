#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include ".build_release/src/caffe/proto/caffe.pb.h"

using namespace std;
using namespace caffe;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

const int kProtoReadBytesLimit = INT_MAX;

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  if (fd == -1) { cout << "File not found: " << filename << endl; return false; }
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

// Compile && run:
// g++ read_proto.cpp .build_release/src/caffe/proto/caffe.pb.cc -lprotobuf -o read_proto && ./read_proto
int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  NetParameter param;
  const char* filename = "models/bvlc_alexnet/bvlc_alexnet.caffemodel";

  if (!ReadProtoFromBinaryFile(filename, &param)) {
      cout << "Failed to parse NetParameter file: " << filename << endl;
      return -1;
  }

  cout << "Network name: " << param.name() << endl;

  int num_source_layers = param.layers_size();
  cout << "Num source layers: " << num_source_layers << endl << endl;
  for (int i = 0; i < num_source_layers; ++i) {
      const V1LayerParameter& source_layer = param.layers(i);
      const string& source_layer_name = source_layer.name();

      cout << "Source layer " << source_layer_name << endl;
      for (int j = 0; j < source_layer.blobs_size(); ++j) {
          const BlobProto& blob = source_layer.blobs(j);

          if (blob.double_data_size() > 0) {
              cout << "Double data: ";
              cout << blob.double_data_size() << endl;
              for (int k = 0; k < blob.double_data_size(); ++k) {
                  //cout << blob.double_data(k) << " ";
              }
          }
          else if (blob.data_size() > 0) {
              cout << "Single data: ";
              cout << blob.data_size() << endl;
              for (int k = 0; k < blob.data_size(); ++k) {
                  //cout << blob.data(k) << " ";
              }
          }
          else if (blob.half_data_size() > 0) {
              cout << "Half data: ";
              cout << blob.half_data_size() << endl;
              for (int k = 0; k < blob.half_data_size(); ++k) {
                  //cout << blob.half_data(k) << " ";
              }
          }
      }
      cout << endl;
  }

  // Optional:  Delete all global objects allocated by libprotobuf.
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}
