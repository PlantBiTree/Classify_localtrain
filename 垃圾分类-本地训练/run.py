from flask_sever import sess
from predict_local import prediction_result_from_img, init_artificial_neural_network

if __name__ == '__main__':
    model = init_artificial_neural_network(sess);
    while True:
        try:
            img_url = input("请输入图片地址:")
            print('您输入的图片地址为：' + img_url)
            res = prediction_result_from_img(model, img_url)
        except Exception as e:
            print('发生了异常：', e)
