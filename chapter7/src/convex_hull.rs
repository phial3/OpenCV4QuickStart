

use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec4i, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let mut img = imgcodecs::imread(&(BASE_PATH.to_owned() + "hand.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 二值化
    let mut gray = Mat::default();
    let mut binary = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    imgproc::threshold(&gray, &mut binary, 105.0, 255.0, imgproc::THRESH_BINARY)?;
    // 开运算消除细小区域
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(3, 3),
        Point::new(-1, -1)
    )?;

    imgproc::morphology_ex(&binary.clone(),
                           &mut binary,
                           imgproc::MORPH_OPEN,
                           &kernel,
                           Point::new(-1, -1),
                           1,
                           opencv::core::BORDER_CONSTANT,
                           Scalar::new(0.0, 0.0, 0.0, 0.0))?;
    highgui::imshow("binary", &binary)?;

    // 轮廓发现
    let mut contours = Vector::<Vector<Point>>::new();
    let mut hierarchy = Vector::<Vec4i>::new();
    imgproc::find_contours_with_hierarchy(
        &binary,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0)
    )?;

    // 遍历每个轮廓
    for n in 0..contours.len() {
        // 计算凸包
        let mut hull = Vector::<Point>::new();
        imgproc::convex_hull(&contours.get(n)?, &mut hull, false, false)?;
        // 绘制凸包
        for i in 0..hull.len() {
            // 绘制凸包顶点
            imgproc::circle(
                &mut img,
                hull.get(i)?,
                4,
                Scalar::new(255.0, 0.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0
            )?;
            // 连接凸包
            if i == hull.len() - 1 {
                imgproc::line(
                    &mut img,
                    hull.get(i)?,
                    hull.get(0)?,
                    Scalar::new(0.0, 0.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0
                )?;
                break;
            }
            imgproc::line(
                &mut img,
                hull.get(i)?,
                hull.get(i + 1)?,
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                imgproc::LINE_8,
                0
            )?;
        }
    }
    highgui::imshow("hull", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}
