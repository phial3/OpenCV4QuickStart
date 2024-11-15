use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Size},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {
    // 读取图像，黑白图像边缘检测结果较为明显
    let img = imgcodecs::imread(&format!("{}{}", BASE_PATH, "equalLena.png"), imgcodecs::IMREAD_ANYDEPTH)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    let mut result = Mat::default();

    // 未滤波提取边缘
    imgproc::laplacian(
        &img,
        &mut result,
        opencv::core::CV_16S,
        3,  // ksize
        1.0,  // scale
        0.0,  // delta
        opencv::core::BORDER_DEFAULT // boder_type: 边界填充类型
    )?;

    let mut result_abs = Mat::default();
    opencv::core::convert_scale_abs(&result, &mut result_abs, 1.0, 0.0)?;

    // 滤波后提取Laplacian边缘
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &img,
        &mut blurred,
        Size::new(3, 3),
        5.0,  // sigmaX
        0.0,  // sigmaY
        opencv::core::BORDER_DEFAULT
    )?;

    let mut blurred_laplacian = Mat::default();
    imgproc::laplacian(
        &blurred,
        &mut blurred_laplacian,
        opencv::core::CV_16S,
        3,  // ksize
        1.0,  // scale
        0.0,  // delta
        opencv::core::BORDER_DEFAULT
    )?;

    let mut blurred_laplacian_abs = Mat::default();
    opencv::core::convert_scale_abs(&blurred_laplacian, &mut blurred_laplacian_abs,1.0, 0.0)?;

    // 显示图像
    highgui::imshow("result", &result_abs)?;
    highgui::imshow("result_laplacian", &blurred_laplacian_abs)?;
    highgui::wait_key(0)?;

    Ok(())
}