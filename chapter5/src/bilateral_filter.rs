use anyhow::{Context, Error, Result};
use opencv::{
    core::{Mat, BorderTypes},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {
    // 读取含有人脸的两张图像
    let img1 = imgcodecs::imread(&(BASE_PATH.to_owned() + "face1.png"), imgcodecs::IMREAD_COLOR)?;
    let img2 = imgcodecs::imread(&(BASE_PATH.to_owned() + "face2.png"), imgcodecs::IMREAD_COLOR)?;
    if img1.empty() || img2.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    let mut result1 = Mat::default();
    let mut result2 = Mat::default();
    let mut result3 = Mat::default();
    let mut result4 = Mat::default();

    // 使用不同滤波器直径对图像进行滤波, border_type=4, BorderTypes::BORDER_DEFAULT
    imgproc::bilateral_filter(&img1, &mut result1, 9, 50.0, 25.0 / 2.0, 4)
        .context("Bilateral filter1 failed").unwrap();
    imgproc::bilateral_filter(&img1, &mut result2, 25, 50.0, 25.0 / 2.0, 4)
        .context("Bilateral filter2 failed").unwrap();

    // 使用不同标准差值对第二张图像进行滤波
    imgproc::bilateral_filter(&img2, &mut result3, 9, 9.0, 9.0, 4)
        .context("Bilateral filter3 failed").unwrap();
    imgproc::bilateral_filter(&img2, &mut result4, 9, 200.0, 200.0, 4)
        .context("Bilateral filter4 failed").unwrap();

    // 显示原始图像
    highgui::named_window("img1", highgui::WINDOW_NORMAL)?;
    highgui::imshow("img1", &img1)?;

    highgui::named_window("img2", highgui::WINDOW_NORMAL)?;
    highgui::imshow("img2", &img2)?;

    // 显示不同直径的滤波结果
    highgui::named_window("result1", highgui::WINDOW_NORMAL)?;
    highgui::imshow("result1", &result1)?;

    highgui::named_window("result2", highgui::WINDOW_NORMAL)?;
    highgui::imshow("result2", &result2)?;

    // 显示不同标准差的滤波结果
    highgui::named_window("result3", highgui::WINDOW_NORMAL)?;
    highgui::imshow("result3", &result3)?;

    highgui::named_window("result4", highgui::WINDOW_NORMAL)?;
    highgui::imshow("result4", &result4)?;

    // 等待按键事件
    highgui::wait_key(0)?;

    Ok(())
}