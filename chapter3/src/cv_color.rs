use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()>{

    // 读取图像
    let img = imgcodecs::imread(&(BASE_PATH.to_string() + "lena.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    // NOTE: 将图像转换为 CV_32F 类型
    // 图像数据的存储格式通常为 CV_8U（8 位无符号整数类型），即每个像素的值在 0 到 255 之间。
    // 这种格式适用于大多数图像操作，比如显示图像和读取图像。然而，某些图像处理操作（例如，图像归一化、滤波、或一些数学运算）
    // 可能需要更高的精度或不同的数值范围，这时就需要将图像转换为浮动精度格式，例如 CV_32F（32 位浮动类型）
    let mut img32 = Mat::default();
    // 实现了将图像从 CV_8U 转换为 CV_32F，并且对像素值进行了归一化：1.0 / 255.0 将像素值缩放到 [0, 1] 范围。
    // 归一化的目的是将图像数据的像素值转换为浮动点数，使其适合于进一步的数值计算（例如，深度学习或其他高精度图像处理任务）。
    img.convert_to(&mut img32, opencv::core::CV_32F, 1.0 / 255.0, 0.0)?;

    // 创建存储不同颜色空间的 Mat 对象
    let mut hsv = Mat::default();
    let mut yuv = Mat::default();
    let mut lab = Mat::default();
    let mut gray = Mat::default();

    // 转换到不同的颜色空间
    imgproc::cvt_color(&img32, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;
    imgproc::cvt_color(&img32, &mut yuv, imgproc::COLOR_BGR2YUV, 0)?;
    imgproc::cvt_color(&img32, &mut lab, imgproc::COLOR_BGR2Lab, 0)?;
    imgproc::cvt_color(&img32, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // 显示不同的图像
    highgui::imshow("原图", &img32)?;
    highgui::imshow("HSV", &hsv)?;
    highgui::imshow("YUV", &yuv)?;
    highgui::imshow("Lab", &lab)?;
    highgui::imshow("gray", &gray)?;

    // 等待键盘输入
    highgui::wait_key(0)?;

    Ok(())
}