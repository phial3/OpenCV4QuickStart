use anyhow::{Context, Result, Error};
use opencv::{
    core::{Mat, Point, Point2d, Point2f, Size},
    highgui,
    imgproc,
    imgcodecs,
    prelude::*,
};
const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}dial.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }


    let mut img1 = Mat::default();
    let mut img2 = Mat::default();

    // 极坐标变换的原点
    let center = Point2f::new(img.cols() as f32 / 2.0, img.rows() as f32 / 2.0);

    // 正极坐标变换
    imgproc::warp_polar(
        &img,
        &mut img1,
        Size::new(300, 600),
        center,
        center.x as f64,
        imgproc::INTER_LINEAR + imgproc::WARP_POLAR_LINEAR,
    )?;

    // 逆极坐标变换
    imgproc::warp_polar(
        &img1,
        &mut img2,
        Size::new(img.cols(), img.rows()),
        center,
        center.x as f64,
        imgproc::INTER_LINEAR + imgproc::WARP_POLAR_LINEAR + imgproc::WARP_INVERSE_MAP,
    )?;

    // 显示图像
    highgui::imshow("原表盘图", &img)?;
    highgui::imshow("表盘极坐标变换结果", &img1)?;
    highgui::imshow("逆变换结果", &img2)?;

    highgui::wait_key(0)?;

    Ok(())
}