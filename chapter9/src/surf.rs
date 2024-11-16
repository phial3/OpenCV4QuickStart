use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Point2f, Scalar, RNG, DMatch, Ptr, Vector, KeyPoint},
    imgcodecs,
    imgproc,
    highgui,
    features2d::{self, Feature2D, SIFT},
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter9/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    // 创建SIFT特征点检测器
    let mut surf = SIFT::create(
        0,             // 关键点阈值
        3,          // 4组金字塔
        10.0,      // 每组金字塔有3层
        10.0,       // 边缘响应阈值
        1.6,               // 初始缩放因子
        false, // 计算方向
    ).context("创建SIFT特征点检测器失败").unwrap();

    // 计算SIFT关键点
    let mut keypoints = Vector::<KeyPoint>::new();
    surf.detect(&img, &mut keypoints, &opencv::core::no_array()).context("计算SIFT关键点失败").unwrap();

    // 计算SIFT描述子
    let mut descriptions = Mat::default();
    surf.compute(&img, &mut keypoints, &mut descriptions).context("计算SIFT描述子失败").unwrap();

    // 绘制特征点
    let mut img_no_angle = Mat::default();
    let mut img_with_angle = img.clone();

    // 绘制不含角度和大小的结果
    features2d::draw_keypoints(
        &img,
        &keypoints,
        &mut img_no_angle,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        features2d::DrawMatchesFlags::DEFAULT,
    )?;

    // 绘制含有角度和大小的结果
    features2d::draw_keypoints(
        &img,
        &keypoints,
        &mut img_with_angle,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        features2d::DrawMatchesFlags::DRAW_RICH_KEYPOINTS,
    )?;

    // 显示结果
    highgui::imshow("不含角度和大小的结果", &img_no_angle)?;
    highgui::imshow("含有角度和大小的结果", &img_with_angle)?;

    highgui::wait_key(0)?;

    Ok(())
}
