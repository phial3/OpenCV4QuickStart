use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec4i, Vec2f, Point2f, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&(BASE_PATH.to_owned() + "HoughLines.jpg"), imgcodecs::IMREAD_GRAYSCALE)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 检测边缘图像，并二值化
    let mut edge = Mat::default();
    imgproc::canny(&img, &mut edge, 80.0, 180.0, 3, false)?;
    imgproc::threshold(&edge.clone(), &mut edge, 170.0, 255.0, imgproc::THRESH_BINARY)?;
    // 利用渐进概率式霍夫变换提取直线
    let mut lines_p1 = Vector::<Vec4i>::new();
    let mut lines_p2 = Vector::<Vec4i>::new();
    // 两个点连接最大距离10
    imgproc::hough_lines_p(
        &edge,
        &mut lines_p1,
        1.0,
        opencv::core::CV_PI / 180.0,
        150,
        30.0,
        10.0
    )?;
    // 两个点连接最大距离30
    imgproc::hough_lines_p(
        &edge,
        &mut lines_p2,
        1.0,
        opencv::core::CV_PI / 180.0,
        150,
        30.0,
        30.0
    )?;
    // 绘制两个点连接最大距离10直线检测结果
    let mut img1 = img.clone();
    for i in 0..lines_p1.len() {
        let line = lines_p1.get(i)?;
        imgproc::line(
            &mut img1,
            Point::new(line[0], line[1]),
            Point::new(line[2], line[3]),
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            3,
            8,
            0
        )?;
    }
    // 绘制两个点连接最大距离30直线检测结果
    let mut img2 = img.clone();
    for i in 0..lines_p2.len() {
        let line = lines_p2.get(i)?;
        imgproc::line(
            &mut img2,
            Point::new(line[0], line[1]),
            Point::new(line[2], line[3]),
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            3,
            8,
            0
        )?;
    }
    // 显示图像
    highgui::imshow("img1", &img1)?;
    highgui::imshow("img2", &img2)?;
    highgui::wait_key(0)?;

    Ok(())
}