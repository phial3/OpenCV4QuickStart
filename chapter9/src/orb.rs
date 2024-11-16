use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Point2f, Scalar, RNG, Vector, KeyPoint},
    imgcodecs,
    imgproc,
    highgui,
    features2d::{self, DrawMatchesFlags, ORB, ORB_ScoreType},
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter9/";

pub(crate) fn run() -> Result<()> {

    // 读取图像
    let mut img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 创建 ORB 特征点检测器
    let mut orb = ORB::create(
        500,                    // 特征点数目
        1.2f32,                // 金字塔层级之间的缩放比例
        8,                      // 金字塔图像层数系数
        31,                     // 边缘阈值
        0,                      // 原图在金字塔中的层数
        2,                      // 生成描述子时需要用的像素点数目
        ORB_ScoreType::HARRIS_SCORE, // 使用 Harris 方法评价特征点
        31,                     // 生成描述子时关键点周围邻域的尺寸
        20,                     // 计算 FAST 角点时像素值差值的阈值
    )?;

    // 检测关键点
    let mut keypoints = Vector::new();
    orb.detect(&img, &mut keypoints, &opencv::core::no_array())?;

    // 计算描述子
    let mut descriptions = Mat::default();
    orb.compute(&img, &mut keypoints, &mut descriptions)?;

    // 准备绘制结果
    let mut img_angle = Mat::default();
    img.copy_to(&mut img_angle)?;

    // 绘制不含角度和大小的结果
    let mut img_simple = Mat::default();
    features2d::draw_keypoints(
        &img,
        &keypoints,
        &mut img_simple,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        DrawMatchesFlags::DEFAULT,
    )?;

    // 绘制含有角度和大小的结果
    let mut img_rich = Mat::default();
    features2d::draw_keypoints(
        &img,
        &keypoints,
        &mut img_rich,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        DrawMatchesFlags::DRAW_RICH_KEYPOINTS,
    )?;

    // 显示结果
    highgui::imshow("不含角度和大小的结果", &img_simple)?;
    highgui::imshow("含有角度和大小的结果", &img_rich)?;
    highgui::wait_key(0)?;

    Ok(())
}
