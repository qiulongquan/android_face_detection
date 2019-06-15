```
脸部识别制作方法

图片准备完成后record文件准备完成后，训练的config文件设定完成后 开始进行实际训练步骤。
# From the tensorflow/models/research/ directory
python object_detection/model_main.py     --pipeline_config_path=object_detection/training/ssd_mobilenet_v1_coco.config    --model_dir=object_detection/training    --num_train_steps=30000    --num_eval_steps=2000    --alsologtostderr

这个是打开tensorboard页面的命令，查看详细的训练执行情况。
Anaconda Prompt 定位到  models\research\object_detection 文件夹下，运行
tensorboard --logdir='training'

训练完成后执行冻结输出pb文件操作
Anaconda Prompt 定位到  models\research\object_detection 文件夹下，运行
python export_inference_graph.py   --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_coco.config  --trained_checkpoint_prefix training/model.ckpt-30000  --output_directory qiulongquan


手机端tflite文件制作方法

PC端pb文件转化为手机端tflite文件方法 先确定已经获取了pb文件
先冻结模型，即将变量用常量替代（上一步frozen的pb模型直接转换会报错，需使用export_tflite_ssd_graph.py进行优化后再转换）。Anaconda Prompt 定位到  models\research\object_detection 文件夹下，运行：
会在输出文件夹中生成2个文件pb和pbtxt，没有生产表示有问题。
python export_tflite_ssd_graph.py --pipeline_config_path=training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix=training/model.ckpt-30000 --output_directory qiulongquan_tflite --add_postprocessing_op=true

生产2个文件后，执行下面的转换
run this from the /Users/qiulongquan/tensorflow/ directory:

bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=/Users/qiulongquan/tensorflow/qiulongquan_tflite/tflite_graph.pb \
--output_file=/Users/qiulongquan/tensorflow/qiulongquan_tflite/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 --std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops \
--default_ranges_min=0 --default_ranges_max=255


注意的地方
1.For a floating point model, run this from the tensorflow/ directory:
2.加入范围最大值和最小值。 最大值我实验了100和255 在检测邱珑泉的脸的实验中没有明显的差别。min数值只能是0开始  --default_ranges_min=0 --default_ranges_max=255
3.推理类型是QUANTIZED_UINT8。—inference_type=QUANTIZED_UINT8



Origin

bazel run -c opt tensorflow/lite/toco:toco -- --input_file=/Users/qiulongquan/tensorflow/qiulongquan_tflite/tflite_graph.pb --output_file=/Users/qiulongquan/tensorflow/qiulongquan_tflite/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops --default_ranges_min= --default_ranges_max=

Fixed

bazel run -c opt tensorflow/lite/toco:toco -- --input_file=/Users/qiulongquan/tensorflow/qiulongquan_tflite/tflite_graph.pb --output_file=/Users/qiulongquan/tensorflow/qiulongquan_tflite/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops --default_ranges_min=0 --default_ranges_max=255


Sample

bazel run -c opt tensorflow/lite/toco:toco -- \ --input_file=/home/username/tensorflow-master/models/research/object_detection/customized_model/path to tflite_graph.pb \ --output_file=/home/username/tensorflow-master/models/research/object_detection/your any folder name/detect.tflite \ --input_shapes=1,300,300,3 \ --input_arrays=normalized_input_image_tensor \ --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \ --inference_type=FLOAT \ --mean_values=128 \ --std_values=128 \ --change_concat_input_ranges=false \ --allow_custom_ops


实际上保存了训练的神经网络的结构图信息。存储格式为protobuffer，所以文件名后缀为pb

安装bazel
https://docs.bazel.build/versions/master/install-os-x.html
Bazel是什么东西， google开源的一个用于多种语言混合的编译器，速度很快
https://www.jianshu.com/p/b2c41344c554
https://docs.bazel.build/versions/master/getting-started.html


安装Android NDK 14b   这个好像不需要
https://developer.android.com/ndk/downloads/older_releases.html#ndk-14b-downloads
android-ndk-r14b-darwin-x86_64.zip


如果在一开始就使用 mobile v2的配置，应该直接就可以使用做好的pb 不用在转换了。需要测试一下

生成冻结后的模型，再转换为对应的TF Lite模型，包括float类型的（模型更大，更准确）和量化后uint8类型的模型（模型更小，但准确率不高）
float32型：

手机端quantized转换只能用在ssd系列faster_rcnn不可以。我估计是因为faster_rcnn容量太大了。
90种物体
ssd_mobilenet_v1_coco_2017_11_17
mscoco_complete_label_map.pbtxt

601种物体
faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12
oid_v4_label_map.pbtxt


下面这2个手机端试一下
quantized的包含tflite frozen graphs (txt/binary). tflite_graph.pbtxt   tflite_graph.pb
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
ssd_mobilenet_v2_quantized_coco

ssdlite_mobilenet_v2_coco



mAP (mean average precision):    平均精度

How to train your own Object Detector with TensorFlow’s Object Detector API
国外的人做的训练集  采用google cloud 200张照片 平均精度80% 损失函数1左右 step 22000  用了1个多小时
https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9


参考资料
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md
https://blog.csdn.net/qq_26535271/article/details/83031412
https://blog.csdn.net/dy_guox/article/details/80139981
https://www.tensorflow.org/install/source


```
