use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec2i, Vector, Point2f, TermCriteria, TermCriteria_Type},
    imgcodecs,
    imgproc,
    highgui,
    features2d,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter9/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    // 转换为灰度图像
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // 提取角点
    let max_corners = 100; // 检测角点数目
    let quality_level = 0.01; // 质量等级
    let min_distance = 0.04; // 两个角点之间的最小欧式距离
    let mut corners = Vector::<Point2f>::new();

    imgproc::good_features_to_track(
        &gray,
        &mut corners,
        max_corners,
        quality_level,
        min_distance,
        &opencv::core::no_array(),
        3,
        false,
        0.04
    )?;

    // 计算亚像素级别角点坐标
    let mut corners_sub = corners.clone(); // 角点备份
    let win_size = Size::new(5, 5);
    let zero_zone = Size::new(-1, -1);
    let criteria = TermCriteria::new(
        TermCriteria_Type::EPS as i32 + TermCriteria_Type::COUNT as i32,
        40,
        0.001
    )?;

    imgproc::corner_sub_pix(
        &gray,
        &mut corners_sub,
        win_size,
        zero_zone,
        criteria
    )?;

    // 输出初始坐标和精细坐标
    for (i, (corner, corner_sub)) in corners.iter().zip(corners_sub.iter()).enumerate() {
        println!(
            "第{}个角点初始坐标：({}, {})   精细后坐标：({}, {})",
            i,
            corner.x, corner.y,
            corner_sub.x, corner_sub.y
        );
    }

    Ok(())
}

