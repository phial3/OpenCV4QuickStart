use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Rect, Vec2i, Vector, RNG},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter8/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    // 绘制矩形
    let mut img_rect = img.clone();
    let rect = Rect::new(80, 30, 340, 390);
    imgproc::rectangle(
        &mut img_rect,
        rect,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        0
    )?;
    highgui::imshow("选择的矩形区域", &img_rect)?;

    // 进行分割
    let mut bgdmod = Mat::zeros(1, 65, opencv::core::CV_64FC1)?.to_mat()?;
    let mut fgdmod = Mat::zeros(1, 65, opencv::core::CV_64FC1)?.to_mat()?;
    let mut mask = Mat::zeros(img.rows(), img.cols(), opencv::core::CV_8UC1)?.to_mat()?;
    imgproc::grab_cut(
        &img,
        &mut mask,
        rect,
        &mut bgdmod,
        &mut fgdmod,
        5,
        imgproc::GC_INIT_WITH_RECT
    )?;

    // 将分割出的前景绘制回来
    let mut result = Mat::default();

    // 处理掩码
    for row in 0..mask.rows() {
        for col in 0..mask.cols() {
            let n = *mask.at_2d::<u8>(row, col)?;
            // 将明显是前景和可能是前景的区域都保留
            *mask.at_2d_mut::<u8>(row, col)? = if n == 1 || n == 3 {
                255
            } else {
                // 将明显是背景和可能是背景的区域都删除
                0
            };
        }
    }

    // 使用掩码提取前景
    opencv::core::bitwise_and(&img, &img, &mut result, &mask)?;

    // 显示结果
    highgui::imshow("分割结果", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}

