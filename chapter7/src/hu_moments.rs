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

    // 转换为灰度图
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    // 二值化
    let mut binary = Mat::default();
    imgproc::threshold(
        &gray,
        &mut binary,
        105.0,
        255.0,
        imgproc::THRESH_BINARY
    )?;

    // 开运算消除细小区域
    let k = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(3, 3),
        Point::new(-1, -1)
    )?;

    // 开运算
    imgproc::morphology_ex(
        &binary.clone(),
        &mut binary,
        imgproc::MORPH_OPEN,
        &k,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        Scalar::default()
    )?;
    // 轮廓发现
    let mut contours = Vector::<Mat>::new();
    let mut hierarchy = Vector::<Vec4i>::new();

    imgproc::find_contours_with_hierarchy(
        &binary,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0)
    )?;
    // 计算 Hu 矩
    for n in 0..contours.len() {
        let contour = contours.get(n)?;

        // 计算矩
        let moments = imgproc::moments(&contour, true)?;

        // 计算 Hu 矩
        let mut hu = [0.0f64; 7];
        imgproc::hu_moments(moments, &mut hu)?;

        // 打印 Hu 矩
        println!("Hu Moments for contour {}: {:?}", n, hu);
    }

    Ok(())
}