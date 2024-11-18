use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use opencv::{
    calib3d,
    core::{Mat, Point, Point2f, Point3f, Scalar, Size, Vector, TermCriteria},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter10/";

pub(crate) fn run() -> Result<()> {
    // 读取所有图像
    let mut imgs: Vec<Mat> = Vec::new();
    let file = File::open(BASE_PATH.to_string() + "calibdata.txt")?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let image_name = line?;
        let img = imgcodecs::imread(&(BASE_PATH.to_string() + &image_name), imgcodecs::IMREAD_COLOR)?;
        imgs.push(img);
    }

    println!("共有 {} 张图像", imgs.len());

    // 输入前文计算得到的内参矩阵
    let camera_matrix = Mat::from_slice(&[
        532.016297, 0.0, 332.172519,
        0.0, 531.565159, 233.388075,
        0.0, 0.0, 1.0,
    ])?.try_clone()?;

    // 输入前文计算得到的畸变系数
    let dist_coeffs = Mat::from_slice(&[
        -0.285188, 0.080097, 0.001274, -0.002415, 0.106579,
    ])?.try_clone()?;

    let mut undist_imgs: Vec<Mat> = Vec::new();
    // let image_size = Size::new(imgs[0].cols(), imgs[0].rows());
    // 使用 initUndistortRectifyMap() 和 remap() 函数校正图像
    // init_undist_and_remap(imgs, camera_matrix, dist_coeffs, image_size, &mut undist_imgs)?;

    // 使用 undistort() 函数直接计算校正图像
    undist(imgs.clone(), camera_matrix, dist_coeffs, &mut undist_imgs)?;

    // 显示校正前后的图像
    for (i, img) in imgs.iter().enumerate() {
        let window_number = i.to_string();
        highgui::imshow(&format!("未校正图像{}", window_number), &img)?;
        highgui::imshow(&format!("校正后图像{}", window_number), &undist_imgs[i])?;
    }

    highgui::wait_key(0)?;

    Ok(())
}

fn init_undist_and_remap(
    imgs: Vec<Mat>,             // 所有原图像向量
    camera_matrix: Mat,         // 相机内参
    dist_coeffs: Mat,           // 相机畸变系数
    image_size: Size,           // 图像的尺寸
    undist_imgs: &mut Vec<Mat>, // 校正后的输出图像
) -> opencv::Result<()> {
    // 计算映射坐标矩阵
    let mut mapx = Mat::default();
    let mut mapy = Mat::default();
    let R = Mat::eye(3, 3, opencv::core::CV_32F)?;

    // 使用 initUndistortRectifyMap() 和 remap() 函数校正图像
    calib3d::init_undistort_rectify_map(
        &camera_matrix,
        &dist_coeffs,
        &R,
        &camera_matrix,
        image_size,
        opencv::core::CV_32FC1,
        &mut mapx,
        &mut mapy,
    )?;

    // 校正图像
    for img in imgs {
        let mut undist_img = Mat::default();
        imgproc::remap(&img, &mut undist_img, &mapx, &mapy, imgproc::INTER_LINEAR, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
        undist_imgs.push(undist_img);
    }

    Ok(())
}


fn undist(
    imgs: Vec<Mat>,             // 所有原图像向量
    camera_matrix: Mat,         // 相机内参
    dist_coeffs: Mat,           // 相机畸变系数
    undist_imgs: &mut Vec<Mat>, // 校正后的输出图像
) -> opencv::Result<()> {
    for img in imgs {
        let mut undist_img = Mat::default();
        // 使用 undistort() 函数直接计算校正图像
        calib3d::undistort(&img, &mut undist_img, &camera_matrix, &dist_coeffs, &opencv::core::no_array()).context("undistort 计算校正图像失败").unwrap();
        undist_imgs.push(undist_img);
    }

    Ok(())
}