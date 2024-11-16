use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Point2f, Scalar, RNG, Vector, KeyPoint},
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

    // 深拷贝用于第二种方法绘制角点
    let mut img2 = Mat::default();
    img.copy_to(&mut img2)?;

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

    // 用于两种不同方式绘制角点
    let mut img_circle = img.clone();
    let mut key_points = Vector::<KeyPoint>::new();

    // 创建随机数生成器
    let mut rng = RNG::default()?;
    rng.set_state(10086); // 设置随机数种子

    // 绘制角点
    for corner in corners {
        // 第一种方式：使用 circle() 函数
        let b = rng.uniform_f32(0.0, 256.0)? as f64;
        let g = rng.uniform_f32(0.0, 256.0)? as f64;
        let r = rng.uniform_f32(0.0, 256.0)? as f64;

        let point = Point2f::new(corner.x, corner.y);
        imgproc::circle(
            &mut img_circle,
            Point::new(point.x as i32, point.y as i32),
            5,
            Scalar::new(b, g, r, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        // 将角点存入 KeyPoint 中，用于第二种绘制方式
        let mut key_point = KeyPoint::default()?;
        key_point.set_pt(point);
        key_points.push(key_point);
    }

    // 第二种方式：使用 draw_keypoints() 函数
    let mut img_keypoints = Mat::default();
    features2d::draw_keypoints(
        &img2,
        &key_points,
        &mut img_keypoints,
        Scalar::new(-1.0, -1.0, -1.0, 0.0), // 默认颜色
        features2d::DrawMatchesFlags::DEFAULT,
    )?;

    // 显示结果
    highgui::imshow("用circle()函数绘制角点结果", &img_circle)?;
    highgui::imshow("通过绘制关键点函数绘制角点结果", &img_keypoints)?;
    highgui::wait_key(0)?;

    Ok(())
}
