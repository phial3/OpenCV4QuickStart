use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec2i, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    // 用四个点表示三角形轮廓
    let mut contour: Vector<Point> = Vector::new();
    contour.push(Point::new(0, 0));
    contour.push(Point::new(10, 0));
    contour.push(Point::new(10, 10));
    contour.push(Point::new(5, 5));

    let area = imgproc::contour_area(&contour, false)?;
    println!("area = {}", area);

    let img = imgcodecs::imread(&(format!("{}coins.jpg", BASE_PATH)), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    highgui::imshow("原图", &img)?;

    let mut gray = Mat::default();
    let mut binary = Mat::default();

    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    imgproc::gaussian_blur(&gray.clone(), &mut gray, Size::new(9, 9), 2.0, 2.0, opencv::core::BORDER_DEFAULT)?;
    imgproc::threshold(&gray, &mut binary, 170.0, 255.0, imgproc::THRESH_BINARY | imgproc::THRESH_OTSU)?;

    // 轮廓检测
    let mut contours: Vector<Mat> = Vector::new();
    imgproc::find_contours(
        &binary,
        &mut contours,
        imgproc::RETR_TREE,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0)
    )?;

    // 输出轮廓面积
    for t in 0..contours.len() {
        let area1 = imgproc::contour_area(&contours.get(t)?, false)?;
        println!("第{}轮廓面积 = {}", t, area1);
    }

    highgui::wait_key(0)?;

    Ok(())
}

