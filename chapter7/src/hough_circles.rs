
use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec4i, Vec3f, Point2f, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let mut img = imgcodecs::imread(&(BASE_PATH.to_owned() + "coins.jpg"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    highgui::imshow("原图", &img)?;

    // 转换为灰度图像
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // 平滑滤波
    imgproc::gaussian_blur(&gray.clone(), &mut gray, Size::new(9, 9), 2.0, 2.0, opencv::core::BORDER_DEFAULT)?;

    // 检测圆形
    let mut circles = Vector::<Vec3f>::new();
    let dp = 2.0;
    let min_dist = 10.0; // 两个圆心之间的最小距离
    let param1 = 100.0; // Canny 边缘检测的较大阈值
    let param2 = 100.0; // 累加器阈值
    let min_radius = 20; // 圆形半径的最小值
    let max_radius = 100; // 圆形半径的最大值
    imgproc::hough_circles(
        &gray,
        &mut circles,
        imgproc::HOUGH_GRADIENT,
        dp,
        min_dist,
        param1,
        param2,
        min_radius,
        max_radius,
    )?;

    // 在图像中标记出圆形
    for i in 0..circles.len() {
        // 使用 get 方法访问元素
        let circle = circles.get(i)?;
        // 读取圆心
        let center = Point::new(circle[0].round() as i32, circle[1].round() as i32);
        // 读取半径
        let radius = circle[2].round() as i32;
        // 绘制圆心
        imgproc::circle(&mut img, center, 3, Scalar::new(0.0, 255.0, 0.0, 0.0), -1, 8, 0)?;
        // 绘制圆
        imgproc::circle(&mut img, center, radius, Scalar::new(0.0, 0.0, 255.0, 0.0), 3, 8, 0)?;
    }
    // 显示结果
    highgui::imshow("圆检测结果", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}