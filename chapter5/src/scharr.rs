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

    // 创建用于存储结果的矩阵
    let mut result_x = Mat::default();
    let mut result_y = Mat::default();
    let mut result_xy = Mat::default();

    // X方向一阶边缘
    imgproc::scharr(
        &img,
        &mut result_x,
        opencv::core::CV_16S,
        1,  // dx
        0,  // dy
        1.0,  // scale
        0.0,  // delta
        opencv::core::BORDER_DEFAULT
    )?;

    let mut result_x_abs = Mat::default();
    opencv::core::convert_scale_abs(&result_x, &mut result_x_abs, 1.0, 0.0)?;

    // Y方向一阶边缘
    imgproc::scharr(
        &img,
        &mut result_y,
        opencv::core::CV_16S,
        0,  // dx
        1,  // dy
        1.0,  // scale
        0.0,  // delta
        opencv::core::BORDER_DEFAULT
    )?;

    let mut result_y_abs = Mat::default();
    opencv::core::convert_scale_abs(&result_y, &mut result_y_abs, 1.0, 0.0)?;

    // 整幅图像的一阶边缘
    opencv::core::add(&result_x_abs, &result_y_abs, &mut result_xy, &Mat::default(), -1)?;

    // 显示图像
    highgui::imshow("resultX", &result_x_abs)?;
    highgui::imshow("resultY", &result_y_abs)?;
    highgui::imshow("resultXY", &result_xy)?;

    highgui::wait_key(0)?;

    Ok(())
}

