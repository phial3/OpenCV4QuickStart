use anyhow::Result;
use opencv::{
    prelude::*,
    core::{Mat, Size, Point, BorderTypes},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {
    // 待卷积矩阵
    let points: [u8; 25] = [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,
    ];

    let img = unsafe { Mat::new_rows_cols_with_data_unsafe_def(5, 5, opencv::core::CV_8UC1, points.as_ptr() as *mut _)? };

    // 卷积模板
    let kernel = Mat::from_slice(&[
        1.0, 2.0, 1.0,
        2.0, 0.0, 2.0,
        1.0, 2.0, 1.0
    ])?.try_clone()?;

    // 卷积模板归一化
    let mut kernel_norm = Mat::default();
    kernel.convert_to(&mut kernel_norm, opencv::core::CV_32F, 1.0 / 12.0, 0.0)?;

    // 未归一化卷积结果和归一化卷积结果
    let mut result = Mat::default();
    let mut result_norm = Mat::default();

    imgproc::filter_2d(
        &img,
        &mut result,
        opencv::core::CV_32F,
        &kernel,
        Point::new(-1, -1),
        2.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    imgproc::filter_2d(
        &img,
        &mut result_norm,
        opencv::core::CV_32F,
        &kernel_norm,
        Point::new(-1, -1),
        2.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    println!("result:{:?}", result);
    println!("result_norm:{:?}", result_norm);

    // 图像卷积
    let lena = imgcodecs::imread(&format!("{}{}", BASE_PATH, "lena.png"), imgcodecs::IMREAD_COLOR)?;
    if lena.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    let mut lena_filter = Mat::default();
    imgproc::filter_2d(
        &lena,
        &mut lena_filter,
        -1,
        &kernel_norm,
        Point::new(-1, -1),
        2.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    highgui::imshow("lena_filter", &lena_filter)?;
    highgui::imshow("lena", &lena)?;
    highgui::wait_key(0)?;

    Ok(())
}