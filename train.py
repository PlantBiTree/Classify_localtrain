import tensorflow as tf
import os
from keras.optimizers import adam
from tools.data_gen import data_flow
from models.resnet50 import ResNet50
from keras.layers import Dense,Dropout,BatchNormalization,GlobalAveragePooling2D
from keras.models import \
    Model
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras import regularizers



# 设备控制台输出配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.compat.v1.Session(config=config)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '99'

# 全局配置文件
tf.app.flags.DEFINE_string('test_data_local', './test_data', '测试图片文件夹')
tf.app.flags.DEFINE_string('data_local', './garbage_classify/train_data', '训练图片文件夹')
tf.app.flags.DEFINE_integer('num_classes', 40, '垃圾分类数目')
tf.app.flags.DEFINE_integer('input_size', 224, '模型输入图片大小')
tf.app.flags.DEFINE_integer('batch_size', 16, '图片批处理大小')
tf.app.flags.DEFINE_float('learning_rate',1e-4, '学习率')
tf.app.flags.DEFINE_integer('max_epochs', 4, '轮次')
tf.app.flags.DEFINE_string('train_local', './output_model', '训练输出文件夹')
tf.app.flags.DEFINE_integer('keep_weights_file_num', 20, '如果设置为-1，则文件保持的最大权重数表示无穷大')
FLAGS = tf.app.flags.FLAGS

## test_acc = 0.78
def add_new_last_layer(base_model,num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5,name='dropout1')(x)
    # x = Dense(1024,activation='relu',kernel_regularizer= regularizers.l2(0.0001),name='fc1')(x)
    # x = BatchNormalization(name='bn_fc_00')(x)
    x = Dense(512,activation='relu',kernel_regularizer= regularizers.l2(0.0001),name='fc2')(x)
    x = BatchNormalization(name='bn_fc_01')(x)
    x = Dropout(0.5,name='dropout2')(x)
    x = Dense(num_classes,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=x)
    return model

    #模型微调
def setup_to_finetune(FLAGS,model,layer_number=149):
    # K.set_learning_phase(0)
    for layer in model.layers[:layer_number]:
        layer.trainable = False
    # K.set_learning_phase(1)
    for layer in model.layers[layer_number:]:
        layer.trainable = True
    # Adam = adam(lr=FLAGS.learning_rate,clipnorm=0.001)
    Adam = adam(lr=FLAGS.learning_rate,decay=0.0005)
    model.compile(optimizer=Adam,loss='categorical_crossentropy',metrics=['accuracy'])

    
    #模型初始化设置
def model_fn(FLAGS):
    # K.set_learning_phase(0)
    # 引入初始化resnet50模型
    base_model = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    for layer in base_model.layers:
        layer.trainable = False
    model = add_new_last_layer(base_model,FLAGS.num_classes)
    model.compile(optimizer="adam",loss = 'categorical_crossentropy',metrics=['accuracy'])
    return model
def train_model(FLAGS):
     # 训练数据构建
    train_sequence, validation_sequence = data_flow(FLAGS.data_local, FLAGS.batch_size,
                                                    FLAGS.num_classes, FLAGS.input_size)
    model = model_fn(FLAGS)
    history_tl = model.fit_generator(
        train_sequence,
        steps_per_epoch = len(train_sequence),
        epochs = FLAGS.max_epochs,
        verbose = 1,
        validation_data = validation_sequence,
        max_queue_size = 10,
        shuffle=True
    )
    #模型微调
    setup_to_finetune(FLAGS,model)
    history_tl = model.fit_generator(
        train_sequence,
        steps_per_epoch = len(train_sequence),
        epochs = FLAGS.max_epochs*5,
        verbose = 1,
        callbacks = [
            ModelCheckpoint('./output_model/best.h5',
                            monitor='val_loss', save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                            patience=10, mode='min'),
            EarlyStopping(monitor='val_loss', patience=10),
            ],
        validation_data = validation_sequence,
        max_queue_size = 10,
        shuffle=True
    )
    print('training done!')

    
if __name__ == "__main__":
    train_model(FLAGS)
