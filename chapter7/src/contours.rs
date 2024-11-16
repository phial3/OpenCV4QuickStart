
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
    let img = imgcodecs::imread(&(format!("{}keys.jpg", BASE_PATH)), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    highgui::imshow("原图", &img)?;

    let mut gray = Mat::default();
    let mut binary = Mat::default();

    // 转化成灰度图
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // 平滑滤波
    imgproc::gaussian_blur(&gray.clone(), &mut gray, Size::new(13, 13), 4.0, 4.0, opencv::core::BORDER_DEFAULT)?;

    // 自适应二值化
    imgproc::threshold(&gray, &mut binary, 170.0, 255.0, imgproc::THRESH_BINARY | imgproc::THRESH_OTSU)?;

    // 轮廓发现与绘制
    let mut contours = Vector::<Vector<Point>>::new();
    let mut hierarchy = Vector::<Vec4i>::new();

    imgproc::find_contours_with_hierarchy(
        &binary,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_TREE,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0)
    )?;

    // 绘制轮廓
    let mut result_img = img.clone();
    for t in 0..contours.len() {
        imgproc::draw_contours(
            &mut result_img,
            &contours,
            t as i32,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            &hierarchy,
            0,
            Point::new(0, 0)
        ).context("绘制轮廓失败").unwrap();
    }

    // 输出轮廓结构描述子
    for i in 0..hierarchy.len() {
        let hier_item = hierarchy.get(i)?;
        println!("{:?}", hier_item);
    }

    // 显示结果
    highgui::imshow("轮廓检测结果", &result_img)?;

    highgui::wait_key(0)?;

    Ok(())
}