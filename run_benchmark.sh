PROG=caffe
if [ "$1" = "fp16" ]; then
    PROG=caffe_fp16
fi

MODEL=bvlc_alexnet
#MODEL=bvlc_googlenet
#MODEL=bvlc_reference_caffenet
#MODEL=bvlc_reference_rcnn_ilsvrc13

sudo LD_LIBRARY_PATH=/usr/local/cuda/lib ./build/tools/$PROG time -model=models/$MODEL/deploy.prototxt -gpu all -iterations 100
