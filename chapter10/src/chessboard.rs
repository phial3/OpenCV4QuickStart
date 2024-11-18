use anyhow::{Result, Error, Context};
use image::open;
use opencv::{
    prelude::*,
    core::{Mat, Point2f, Point3f, Scalar, Size, Vector, TermCriteria},
    imgcodecs,
    imgproc,
    highgui,
    calib3d,
    features2d::{SimpleBlobDetector, SimpleBlobDetector_Params, Feature2D},
};

const BASE_PATH: &str = "../data/chapter10/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let mut img1 = imgcodecs::imread(&format!("{}left01.jpg", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    let mut img2 = imgcodecs::imread(&format!("{}circle.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    // 检查图像是否正确加载
    if img1.empty() || img2.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    // 转换为灰度图
    let mut gray1 = Mat::default();
    let mut gray2 = Mat::default();
    imgproc::cvt_color(&img1, &mut gray1, imgproc::COLOR_BGR2GRAY, 0)?;
    imgproc::cvt_color(&img2, &mut gray2, imgproc::COLOR_BGR2GRAY, 0)?;
    println!("gray1 转换为灰度图：{:?}, 类型：{:?}, 通道：{:?}", gray1.size(), gray1.typ(), gray1.channels());
    println!("gray2 转换为灰度图：{:?}, 类型：{:?}, 通道：{:?}", gray2.size(), gray2.typ(), gray2.channels());

    // 定义数目尺寸
    let board_size1 = Size::new(9, 6);  // 方格标定板内角点数目（行，列）
    let board_size2 = Size::new(7, 7);  // 圆形标定板圆心数目（行，列）

    // 检测角点
    let mut img1_points = Vector::<Point2f>::new();
    let mut img2_points = Vector::<Point2f>::new();

    // 计算方格标定板角点
    calib3d::find_chessboard_corners(
        &gray1,
        board_size1,
        &mut img1_points,
        calib3d::CALIB_CB_ADAPTIVE_THRESH
    ).context("方格标定板角点检测失败").unwrap();

    // 计算圆形标定板检点
   unsafe {
       calib3d::find_circles_grid_1(
           &gray2,
           board_size2,
           &mut img2_points,
           calib3d::CALIB_CB_SYMMETRIC_GRID,
           None,
       ).context("圆形标定板角点检测失败").unwrap();
   }

    // 细化角点坐标
    let criteria = TermCriteria::new(
        opencv::core::TermCriteria_EPS + opencv::core::TermCriteria_MAX_ITER,
        30,
        0.1
    )?;

    // 细化方格标定板角点坐标
    imgproc::corner_sub_pix(
        &gray1,
        &mut img1_points,
        Size::new(5, 5),
        Size::new(-1, -1),
        criteria.clone()
    ).context("方格标定板角点细化失败").unwrap();

    // 细化圆形标定板角点坐标
    imgproc::corner_sub_pix(
        &gray2,
        &mut img2_points,
        Size::new(5, 5),
        Size::new(-1, -1),
        criteria
    ).context("圆形标定板角点细化失败").unwrap();

    // 绘制角点检测结果
    calib3d::draw_chessboard_corners(
        &mut img1,
        board_size1,
        &img1_points,
        true
    )?;
    calib3d::draw_chessboard_corners(
        &mut img2,
        board_size2,
        &img2_points,
        true
    )?;

    // 显示结果
    highgui::named_window("方形标定板角点检测结果", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("圆形标定板角点检测结果", highgui::WINDOW_AUTOSIZE)?;
    highgui::imshow("方形标定板角点检测结果", &img1)?;
    highgui::imshow("圆形标定板角点检测结果", &img2)?;

    highgui::wait_key(0)?;

    Ok(())
}
