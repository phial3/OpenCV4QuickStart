use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Size},
    imgcodecs,
    imgproc,
    highgui,
};

pub(crate) fn run() -> Result<()> {
    // 存放分离的Sobel算子
    let mut sobel_x1 = Mat::default();
    let mut sobel_y1 = Mat::default();
    let mut sobel_x2 = Mat::default();
    let mut sobel_y2 = Mat::default();
    let mut sobel_x3 = Mat::default();
    let mut sobel_y3 = Mat::default();

    // 存放分离的Scharr算子
    let mut scharr_x = Mat::default();
    let mut scharr_y = Mat::default();

    // 存放最终算子
    let mut sobel_x1_reshaped = Mat::default();
    let mut sobel_x2_reshaped = Mat::default();
    let mut sobel_x3_reshaped = Mat::default();
    let mut scharr_x_reshaped = Mat::default();

    let mut sobel_x1_final = Mat::default();
    let mut sobel_x2_final = Mat::default();
    let mut sobel_x3_final = Mat::default();
    let mut scharr_x_final = Mat::default();

    // 一阶X方向Sobel算子
    imgproc::get_deriv_kernels(
        &mut sobel_x1,
        &mut sobel_y1,
        1,
        0,
        3,
        false,
        opencv::core::CV_32F,
    ).context("Failed to get Sobel X1 kernels").unwrap();

    sobel_x1.convert_to(&mut sobel_x1_reshaped, opencv::core::CV_8U, 1.0, 0.0)?;
    sobel_x1_reshaped = sobel_x1_reshaped.reshape(1, 0).unwrap().try_clone()?;
    opencv::core::multiply(&sobel_y1, &sobel_x1_reshaped, &mut sobel_x1_final, 1.0, opencv::core::CV_32F)
        .context("Failed to multiply Sobel X1").unwrap();

    println!("sobel_x1: {:?}", sobel_x1);

    // 二阶X方向Sobel算子
    imgproc::get_deriv_kernels(
        &mut sobel_x2,
        &mut sobel_y2,
        2,
        0,
        5,
        false,
        opencv::core::CV_32F,
    )?;

    sobel_x2.convert_to(&mut sobel_x2_reshaped, opencv::core::CV_8U, 1.0, 0.0)?;
    sobel_x2_reshaped = sobel_x2_reshaped.reshape(1, 0).unwrap().try_clone()?;
    opencv::core::multiply(&sobel_y2, &sobel_x2_reshaped, &mut sobel_x2_final, 1.0, opencv::core::CV_32F)
        .context("Failed to multiply Sobel X2").unwrap();

    println!("sobel_x2: {:?}", sobel_x2);

    // 三阶X方向Sobel算子
    imgproc::get_deriv_kernels(
        &mut sobel_x3,
        &mut sobel_y3,
        3,
        0,
        7,
        false,
        opencv::core::CV_32F,
    )?;

    sobel_x3.convert_to(&mut sobel_x3_reshaped, opencv::core::CV_8U, 1.0, 0.0)?;
    sobel_x3_reshaped = sobel_x3_reshaped.reshape(1, 0).unwrap().try_clone()?;
    opencv::core::multiply(&sobel_y3, &sobel_x3_reshaped, &mut sobel_x3_final, 1.0, opencv::core::CV_32F)
        .context("Failed to multiply Sobel X3").unwrap();

    println!("sobel_x3: {:?}", sobel_x3);

    // X方向Scharr算子
    imgproc::get_deriv_kernels(
        &mut scharr_x,
        &mut scharr_y,
        1,
        0,
        imgproc::FILTER_SCHARR,
        false,
        opencv::core::CV_32F,
    )?;

    scharr_x.convert_to(&mut scharr_x_reshaped, opencv::core::CV_8U, 1.0, 0.0)?;
    scharr_x_reshaped = scharr_x_reshaped.reshape(1, 0).unwrap().try_clone()?;
    opencv::core::multiply(&scharr_y, &scharr_x_reshaped, &mut scharr_x_final, 1.0, opencv::core::CV_32F)
        .context("Failed to multiply Scharr X").unwrap();

    // 输出结果
    println!("X方向一阶Sobel算子:{:?}", sobel_x1_final);
    println!("X方向二阶Sobel算子:{:?}", sobel_x2_final);
    println!("X方向三阶Sobel算子:{:?}", sobel_x3_final);
    println!("X方向Scharr算子:{:?}", scharr_x_final);

    Ok(())
}