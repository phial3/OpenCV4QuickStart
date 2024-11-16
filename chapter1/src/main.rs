use anyhow::Result;
use opencv::{
    prelude::*,
    core::{Vector, Vec4b},
    imgproc,
    imgcodecs,
    highgui,
};

const BASE_PATH: &str = "../data/chapter1/";

fn main() -> Result<()> {
    // 1. 读写图像
    // image_read_write();

    // 2. 图像透明度
    image_alpha()?;

    Ok(())
}

fn image_read_write() -> Result<()> {
    let img = imgcodecs::imread(format!("{}{}", BASE_PATH, "lena.png").as_str(), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("Failed to load image");
    }

    // 打印下图片信息
    println!("Image size: {}x{}", img.cols(), img.rows());
    println!("Image channels: {}", img.channels());
    println!("Image depth: {}", img.depth());

    // 显示图片
    highgui::named_window("Lena", 0)?;
    highgui::imshow("Lena", &img)?;
    highgui::wait_key(0)?;

    // 对图像做一些处理（例如，转为灰度图像）
    let mut gray_image = Mat::default();
    imgproc::cvt_color(&img, &mut gray_image, imgproc::COLOR_BGR2GRAY, 0)?;

    // 保存图像
    let mut params = Vector::new();
    params.push(imgcodecs::IMWRITE_PNG_COMPRESSION); //PNG格式图像压缩标志
    params.push(9);  //设置最高压缩质量
    imgcodecs::imwrite("lena_gray.png", &gray_image, &params)?;

    // 这是 def 方式
    // imgcodecs::imwrite_def("lena_gray.png", &gray_image)?;

    Ok(())
}

// 图像透明度
fn image_alpha() -> Result<()> {
    let mut image = unsafe { Mat::new_rows_cols(480, 640, opencv::core::CV_8UC4)? };
    alpha_mat(&mut image).expect("alpha_mat failed.");

    let mut compression_params = Vector::new();
    compression_params.push(imgcodecs::IMWRITE_PNG_COMPRESSION);  //PNG格式图像压缩标志
    compression_params.push(9);                               //设置最高压缩质量
    imgcodecs::imwrite("alpha.png", &image, &compression_params)?;

    Ok(())
}

/// 1、作用：
/// 这个函数通过以下方式修改图像中的每个像素的颜色通道和 alpha 通道：
/// 蓝色通道 (B) 被设为 255（最大值），即图像的蓝色部分是完全饱和的。
/// 绿色通道 (G) 根据图像的列位置进行线性渐变，从左到右逐渐增加。
/// 红色通道 (R) 根据图像的行位置进行线性渐变，从上到下逐渐增加。
/// Alpha 通道 (A) 根据绿色和红色通道的平均值调整透明度，形成透明度的渐变效果
/// 2、效果:
/// 结果图像的 左侧和上方 将呈现 蓝色和绿色的渐变，因为蓝色通道是固定的，而绿色和红色通道随着位置的变化产生渐变。
/// 图像的 右侧和下方 的绿色和红色值逐渐增加，因此颜色渐变变得更加明显。
/// Alpha 通道 的渐变会使得图像的透明度从左上角到右下角发生变化，产生一个平滑的透明度过渡。
fn alpha_mat(img: &mut Mat) -> Result<()> {
    // 确保输入图像是 4 通道（BGRA）
    assert_eq!(img.channels(), 4, "Input image must have 4 channels (BGRA)");

    let rows = img.rows();
    let cols = img.cols();
    // 遍历每个像素并修改颜色通道
    for i in 0..rows {
        for j in 0..cols {
            let mut bgra = img.at_2d_mut::<Vec4b>(i, j)?;

            // 蓝色通道
            // 这意味着所有像素的蓝色通道都将被设为最大值（即纯蓝色, 255）。
            bgra[0] = 255;

            // 绿色通道，按列位置计算
            // 这意味着图像的绿色通道值从左到右逐渐增加（渐变效果）。
            bgra[1] = (((cols - j) as f32 / cols as f32) * 255.0) as u8;

            // 红色通道，按行位置计算
            // 图像的红色通道值从上到下逐渐增加（渐变效果）。
            bgra[2] = (((rows - i) as f32 / rows as f32) * 255.0) as u8;

            // Alpha通道，取红色和绿色通道的平均值
            // alpha 通道的值是绿色和红色通道的平均值的 50%。通过这种方式，透明度（alpha）值根据绿色和红色通道的值来决定，
            // 使得上下和左右方向的透明度渐变效果同时受到绿色和红色通道值的影响。
            bgra[3] = (0.5 * (bgra[1] as f32 + bgra[2] as f32)) as u8
        }
    }

    Ok(())
}