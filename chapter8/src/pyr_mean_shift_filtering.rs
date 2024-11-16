use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Rect, TermCriteria, TermCriteria_Type},
    highgui,
    imgcodecs,
    imgproc,
    photo,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter8/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}coins.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    // 创建结果矩阵
    let mut result1 = Mat::default();
    let mut result2 = Mat::default();

    // 设置终止条件
    let term_criteria = TermCriteria::new(
        TermCriteria_Type::COUNT as i32 + TermCriteria_Type::EPS as i32,
        10,
        0.1
    )?;

    // 第一次分割
    imgproc::pyr_mean_shift_filtering(
        &img,
        &mut result1,
        20.0,
        40.0,
        2,
        term_criteria
    )?;

    // 第二次分割
    imgproc::pyr_mean_shift_filtering(
        &result1,
        &mut result2,
        20.0,
        40.0,
        2,
        term_criteria
    )?;

    // 显示分割结果
    highgui::imshow("img", &img)?;
    highgui::imshow("result1", &result1)?;
    highgui::imshow("result2", &result2)?;

    // 创建Canny边缘检测结果矩阵
    let mut img_canny = Mat::default();
    let mut result1_canny = Mat::default();
    let mut result2_canny = Mat::default();

    // 执行Canny边缘检测
    imgproc::canny(
        &img,
        &mut img_canny,
        150.0,
        300.0,
        3,
        false
    )?;
    imgproc::canny(
        &result1,
        &mut result1_canny,
        150.0,
        300.0,
        3,
        false
    )?;
    imgproc::canny(
        &result2,
        &mut result2_canny,
        150.0,
        300.0,
        3,
        false
    )?;

    // 显示边缘检测结果
    highgui::imshow("imgCanny", &img_canny)?;
    highgui::imshow("result1Canny", &result1_canny)?;
    highgui::imshow("result2Canny", &result2_canny)?;

    highgui::wait_key(0)?;

    Ok(())
}

