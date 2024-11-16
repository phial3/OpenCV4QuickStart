use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec4i, Point2f, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&(BASE_PATH.to_owned() + "approx.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 二值化
    let mut gray = Mat::default();
    let mut binary = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    imgproc::threshold(&gray, &mut binary, 105.0, 255.0, imgproc::THRESH_BINARY)?;

    // 开运算消除细小区域
    let k = imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(3, 3), Point::new(-1, -1))?;
    imgproc::morphology_ex(&binary.clone(), &mut binary, imgproc::MORPH_OPEN, &k, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::default())?;

    // 轮廓发现
    let mut contours = Vector::<Mat>::new();
    let mut hierarchy = Vector::<Vec4i>::new();
    imgproc::find_contours_with_hierarchy(&binary, &mut contours, &mut hierarchy, imgproc::RETR_TREE, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

    for n in 0..contours.len() {
        let contour = contours.get(n)?;
        let moments = imgproc::moments(&contour, true)?;

        // 输出空间矩
        println!("spatial moments:");
        println!("m00: {}", moments.m00);
        println!("m01: {}", moments.m01);
        println!("m10: {}", moments.m10);
        println!("m11: {}", moments.m11);
        println!("m02: {}", moments.m02);
        println!("m20: {}", moments.m20);
        println!("m12: {}", moments.m12);
        println!("m21: {}", moments.m21);
        println!("m03: {}", moments.m03);
        println!("m30: {}", moments.m30);

        // 输出中心矩
        println!("central moments:");
        println!("mu20: {}", moments.mu20);
        println!("mu02: {}", moments.mu02);
        println!("mu11: {}", moments.mu11);
        println!("mu30: {}", moments.mu30);
        println!("mu21: {}", moments.mu21);
        println!("mu12: {}", moments.mu12);
        println!("mu03: {}", moments.mu03);

        // 输出中心归一化矩
        println!("central normalized moments:");
        println!("nu20: {}", moments.nu20);
        println!("nu02: {}", moments.nu02);
        println!("nu11: {}", moments.nu11);
        println!("nu30: {}", moments.nu30);
        println!("nu21: {}", moments.nu21);
        println!("nu12: {}", moments.nu12);
        println!("nu03: {}", moments.nu03);
    }

    Ok(())
}