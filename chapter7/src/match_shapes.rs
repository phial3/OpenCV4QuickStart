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
    let mut img = imgcodecs::imread(&(BASE_PATH.to_owned() + "ABC.png"), imgcodecs::IMREAD_COLOR)?;
    let mut img_b = imgcodecs::imread(&(BASE_PATH.to_owned() + "B.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() || img_b.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 缩放图像
    let mut img_b_resized = Mat::default();
    imgproc::resize(
        &img_b,
        &mut img_b_resized,
        Size::default(),
        0.5,
        0.5,
        imgproc::INTER_LINEAR
    )?;
    // 保存缩放后的图像
    imgcodecs::imwrite(&(BASE_PATH.to_owned() + "B_.png"), &img_b_resized, &Vector::default())?;

    // 显示缩放后的图像
    highgui::imshow("B", &img_b_resized)?;

    // 轮廓提取
    let contours1 = find_contours(&img)?;
    let contours2 = find_contours(&img_b_resized)?;
    // hu矩计算
    let moments2 = imgproc::moments(&contours2.get(0)?, true)?;
    let mut hu2 = [0.0f64; 7];
    imgproc::hu_moments(moments2, &mut hu2)?;

    // 轮廓匹配
    for n in 0..contours1.len() {
        let moments = imgproc::moments(&contours1.get(n)?, true)?;
        let mut hum = [0.0f64; 7];
        imgproc::hu_moments(moments, &mut hum)?;

        // Hu矩匹配
        let dist = imgproc::match_shapes(
            &Vector::from_slice(&hum),
            &Vector::from_slice(&hum),
            imgproc::CONTOURS_MATCH_I1,
            0.0)?;

        if dist < 1.0 {
            imgproc::draw_contours(
                &mut img,
                &contours1,
                n as i32,
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                3,
                imgproc::LINE_8,
                &opencv::core::no_array(),
                i32::MAX,
                Point::new(0, 0)
            )?;
        }
    }
    // 显示结果
    highgui::imshow("match result", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}

fn find_contours(image: &Mat) -> Result<Vector<Mat>> {
    let mut gray = Mat::default();
    let mut binary = Mat::default();
    let mut hierarchy = Vector::<Vec4i>::new();
    let mut contours = Vector::<Mat>::new();
    // 图像灰度化
    imgproc::cvt_color(
        image,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0
    )?;
    // 图像二值化
    imgproc::threshold(
        &gray,
        &mut binary,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU
    )?;
    // 寻找轮廓
    imgproc::find_contours_with_hierarchy(
        &binary,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0)
    )?;

    Ok(contours)
}