use anyhow::Result;
use opencv::{
    prelude::*,
    core::{Mat, Size},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {

    let equal_lena = imgcodecs::imread(&format!("{}{}", BASE_PATH, "equalLena.png"), imgcodecs::IMREAD_ANYDEPTH)?;
    let equal_lena_gauss = imgcodecs::imread(&format!("{}{}", BASE_PATH, "equalLena_gauss.png"), imgcodecs::IMREAD_ANYDEPTH)?;
    let equal_lena_salt = imgcodecs::imread(&format!("{}{}", BASE_PATH, "equalLena_salt.png"), imgcodecs::IMREAD_ANYDEPTH)?;
    if equal_lena.empty() || equal_lena_gauss.empty() || equal_lena_salt.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 存放不同噪声和滤波器尺寸的滤波结果
    let mut result_5 = Mat::default();
    let mut result_9 = Mat::default();
    let mut result_5_gauss = Mat::default();
    let mut result_9_gauss = Mat::default();
    let mut result_5_salt = Mat::default();
    let mut result_9_salt = Mat::default();

    // 高斯模糊
    imgproc::gaussian_blur(
        &equal_lena,
        &mut result_5,
        Size::new(5, 5),
        10.0,
        20.0,
        opencv::core::BORDER_DEFAULT
    )?;

    imgproc::gaussian_blur(
        &equal_lena,
        &mut result_9,
        Size::new(9, 9),
        10.0,
        20.0,
        opencv::core::BORDER_DEFAULT
    )?;

    imgproc::gaussian_blur(
        &equal_lena_gauss,
        &mut result_5_gauss,
        Size::new(5, 5),
        10.0,
        20.0,
        opencv::core::BORDER_DEFAULT
    )?;

    imgproc::gaussian_blur(
        &equal_lena_gauss,
        &mut result_9_gauss,
        Size::new(9, 9),
        10.0,
        20.0,
        opencv::core::BORDER_DEFAULT
    )?;

    imgproc::gaussian_blur(
        &equal_lena_salt,
        &mut result_5_salt,
        Size::new(5, 5),
        10.0,
        20.0,
        opencv::core::BORDER_DEFAULT
    )?;

    imgproc::gaussian_blur(
        &equal_lena_salt,
        &mut result_9_salt,
        Size::new(9, 9),
        10.0,
        20.0,
        opencv::core::BORDER_DEFAULT
    )?;

    // 显示不含噪声图像
    highgui::imshow("equalLena", &equal_lena)?;
    highgui::imshow("result_5", &result_5)?;
    highgui::imshow("result_9", &result_9)?;

    // 显示含有高斯噪声图像
    highgui::imshow("equalLena_gauss", &equal_lena_gauss)?;
    highgui::imshow("result_5gauss", &result_5_gauss)?;
    highgui::imshow("result_9gauss", &result_9_gauss)?;

    // 显示含有椒盐噪声图像
    highgui::imshow("equalLena_salt", &equal_lena_salt)?;
    highgui::imshow("result_5salt", &result_5_salt)?;
    highgui::imshow("result_9salt", &result_9_salt)?;

    highgui::wait_key(0)?;

    Ok(())
}