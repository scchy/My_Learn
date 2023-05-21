// 形状调整
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

int main()
{
    cv::Mat cat_img = cv::imread("./media/cat.jpg");

    // ==== resize ===
    cv::Mat cat_resize;
    cv::resize(cat_img, cat_resize, cv::Size(320, 240));
    cv::imwrite("./output/3.cat_resize.jpg", cat_resize);
    // ======== copy ========
    cv::Mat copy;
    cat_img.copyTo(copy);
    cv::imwrite("./output/3.copy.jpg", copy);

    // ======== ROI裁剪 ========
    cv::Rect rect(100, 100, 200, 100); // x, y, width, height
    cv::Mat roi = cat_img(rect);
    cv::imwrite("./output/3.roi.jpg", roi);

    // ======== 拼接 ========
    cv::Mat dog_img = cv::imread("./media/dog.jpg");
    cv::Mat dog_resize;
    cv::resize(dog_img, dog_resize, cv::Size(320, 240));

    // 水平cat, row - hight一致
    cv::Mat hcnt_img;
    cv::hconcat(cat_resize, dog_resize, hcnt_img);
    cv::imwrite("./output/3.hconcat.jpg", hcnt_img);

    // 或者以vector方式
    std::vector<cv::Mat> imgs {dog_resize, cat_resize, dog_resize, cat_resize};
    cv::Mat hcnt_img2;
    cv::hconcat(imgs, hcnt_img2);
    cv::imwrite("./output/3.hconcat2.jpg", hcnt_img2);

    // 数组方式
    cv::Mat imgs_arr[] = {dog_resize, cat_resize, dog_resize, cat_resize};
    cv::Mat hconcat_img3;
    cv::hconcat(imgs_arr, 4, hconcat_img3); // 4是数组长度
    cv::imwrite("./output/3.hconcat3.jpg", hconcat_img3);

    // 垂直拼接，需要保证两张图片的宽度（cols）一致
    cv::Mat vconcat_img;
    cv::vconcat(cat_resize, dog_resize, vconcat_img);
    cv::imwrite("./output/3.vconcat.jpg", vconcat_img);

    // ======== 翻转 ========
    cv::Mat flip;
    // 三个参数分别是输入图像、输出图像、翻转方向
    // 1-水平翻转 0-垂直翻转 -1 水平垂直翻转
    cv::flip(cat_img, flip, 1); 
    cv::imwrite("./output/3.flip.jpg", flip);

    // ======== 旋转 ========
    cv::Mat rotate;
    cv::rotate(cat_img, rotate, cv::ROTATE_90_CLOCKWISE); // 顺时针旋转90度
    cv::imwrite("./output/3.rotate.jpg", rotate);
 
    return 0;
}