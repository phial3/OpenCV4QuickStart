use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, RNG, Vector, KeyPoint},
    imgcodecs,
    imgproc,
    highgui,
    features2d,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter9/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let mut img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("读取图像错误，请确认图像文件是否正确");
        return Ok(());
    }

    // 转换为灰度图像
    let mut img_gray = Mat::default();
    imgproc::cvt_color(&img, &mut img_gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // 生成随机关键点
    let mut keypoints = Vector::<KeyPoint>::new();
    let mut rng = RNG::default()?;
    rng.set_state(10086); // 设置随机数种子

    for _ in 0..100 {
        let pty = rng.uniform_f32(0.0, (img.rows() - 1) as f32)?;
        let ptx = rng.uniform_f32(0.0, (img.cols() - 1) as f32)?;

        let mut keypoint = KeyPoint::default()?;
        keypoint.pt().x = ptx;
        keypoint.pt().y = pty;
        keypoints.push(keypoint);
    }

    // 绘制关键点
    let mut img_with_kp = Mat::default();
    let mut img_gray_with_kp = Mat::default();

    features2d::draw_keypoints(
        &img,
        &keypoints,
        &mut img_with_kp,
        Scalar::new(0.0, 0.0, 0.0, 0.0), // 黑色
        features2d::DrawMatchesFlags::DEFAULT,
    )?;

    features2d::draw_keypoints(
        &img_gray,
        &keypoints,
        &mut img_gray_with_kp,
        Scalar::new(255.0, 255.0, 255.0, 0.0), // 白色
        features2d::DrawMatchesFlags::DEFAULT,
    )?;

    // 显示结果
    highgui::imshow("img", &img_with_kp)?;
    highgui::imshow("imgGray", &img_gray_with_kp)?;
    highgui::wait_key(0)?;

    Ok(())
}
