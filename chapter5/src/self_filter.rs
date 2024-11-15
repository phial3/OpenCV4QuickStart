use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Size, Point},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {
    let points: [f32; 25] = [
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0
    ];

    let data = unsafe { Mat::new_rows_cols_with_data_unsafe_def(5, 5, opencv::core::CV_32FC1, points.as_ptr() as *mut _)? };

    // X方向、Y方向和联合滤波器的构建
    let a =  unsafe { Mat::new_rows_cols_with_data_unsafe_def( 3, 1, opencv::core::CV_32F, [-1.0, 3.0, -1.0].as_ptr() as *mut _)? };

    let b = a.reshape(1, 0).unwrap().try_clone()?;

    let mut ab = Mat::default();
    opencv::core::multiply(&a, &b, &mut ab, 1.0, opencv::core::CV_32F)?;

    // 验证高斯滤波的可分离性
    let gauss_x = imgproc::get_gaussian_kernel(3, 1.0, opencv::core::CV_32F)?;
    let mut gauss_data = Mat::default();
    let mut gauss_data_xy = Mat::default();

    imgproc::gaussian_blur(
        &data,
        &mut gauss_data,
        Size::new(3, 3),
        1.0,
        1.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    imgproc::sep_filter_2d(
        &data,
        &mut gauss_data_xy,
        -1,
        &gauss_x,
        &gauss_x,
        Point::new(-1, -1),
        0.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    // 输入两种高斯滤波的计算结果
    println!("gaussData={:?}", gauss_data);
    println!("gaussDataXY={:?}", gauss_data_xy);

    // 线性滤波的可分离性
    let mut data_yx = Mat::default();
    let mut data_y = Mat::default();
    let mut data_xy = Mat::default();
    let mut data_xy_sep = Mat::default();

    imgproc::filter_2d(
        &data,
        &mut data_y,
        -1,
        &a,
        Point::new(-1, -1),
        0.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    imgproc::filter_2d(
        &data_y,
        &mut data_yx,
        -1,
        &b,
        Point::new(-1, -1),
        0.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    imgproc::filter_2d(
        &data,
        &mut data_xy,
        -1,
        &ab,
        Point::new(-1, -1),
        0.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    imgproc::sep_filter_2d(
        &data,
        &mut data_xy_sep,
        -1,
        &b,
        &b,
        Point::new(-1, -1),
        0.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    // 输出分离滤波和联合滤波的计算结果
    println!("dataY={:?}", data_y);
    println!("dataYX={:?}", data_yx);
    println!("dataXY={:?}", data_xy);
    println!("dataXY_sep={:?}", data_xy_sep);

    // 对图像的分离操作
    let img = imgcodecs::imread(&format!("{}{}", BASE_PATH, "lena.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    let mut img_yx = Mat::default();
    let mut img_y = Mat::default();
    let mut img_xy = Mat::default();

    imgproc::filter_2d(
        &img,
        &mut img_y,
        -1,
        &a,
        Point::new(-1, -1),
        0.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    imgproc::filter_2d(
        &img_y,
        &mut img_yx,
        -1,
        &b,
        Point::new(-1, -1),
        0.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    imgproc::filter_2d(
        &img,
        &mut img_xy,
        -1,
        &ab,
        Point::new(-1, -1),
        0.0,
        opencv::core::BORDER_CONSTANT,
    )?;

    highgui::imshow("img", &img)?;
    highgui::imshow("imgY", &img_y)?;
    highgui::imshow("imgYX", &img_yx)?;
    highgui::imshow("imgXY", &img_xy)?;

    highgui::wait_key(0)?;

    Ok(())
}

