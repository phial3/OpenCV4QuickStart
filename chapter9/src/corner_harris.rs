use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec2i, Vector, KeyPoint},
    imgcodecs,
    imgproc,
    highgui,
    features2d,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter9/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    // 转换为灰度图像
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // 计算 Harris 系数
    let mut harris = Mat::default();
    let block_size = 2; // 邻域半径
    let aperture_size = 3;
    imgproc::corner_harris(
        &gray,
        &mut harris,
        block_size,
        aperture_size,
        0.04,
        opencv::core::BORDER_DEFAULT,
    )?;

    // 归一化处理
    let mut harris_n = Mat::default();
    opencv::core::normalize(
        &harris,
        &mut harris_n,
        0.0,
        255.0,
        opencv::core::NORM_MINMAX,
        opencv::core::CV_8U,
        &opencv::core::no_array(),
    )?;

    // 寻找 Harris 角点
    let mut key_points = Vector::new();
    for row in 0..harris_n.rows() {
        for col in 0..harris_n.cols() {
            let r = *harris_n.at_2d::<u8>(row, col)?;
            if r > 125 {
                let mut key_point = KeyPoint::default()?;
                key_point.pt().x = col as f32;
                key_point.pt().y = row as f32;
                key_points.push(key_point);
            }
        }
    }

    // 绘制角点
    let mut output = img.clone();
    features2d::draw_keypoints(
        &img,
        &key_points,
        &mut output,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        features2d::DrawMatchesFlags::DEFAULT,
    )?;

    // 显示结果
    highgui::imshow("系数矩阵", &harris_n)?;
    highgui::imshow("Harris角点", &output)?;
    highgui::wait_key(0)?;

    Ok(())
}