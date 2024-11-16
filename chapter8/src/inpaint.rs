use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size},
    highgui,
    imgcodecs,
    imgproc,
    photo,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter8/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img1 = imgcodecs::imread(&format!("{}inpaint1.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    let img2 = imgcodecs::imread(&format!("{}inpaint2.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    // 检查图像是否正确加载
    if img1.empty() || img2.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 显示原始图像
    highgui::imshow("img1", &img1)?;
    highgui::imshow("img2", &img2)?;

    // 转换为灰度图
    let mut img1_gray = Mat::default();
    let mut img2_gray = Mat::default();
    imgproc::cvt_color(&img1, &mut img1_gray, imgproc::COLOR_RGB2GRAY, 0)?;
    imgproc::cvt_color(&img2, &mut img2_gray, imgproc::COLOR_RGB2GRAY, 0)?;

    // 通过阈值处理生成Mask掩模
    let mut img1_mask = Mat::default();
    let mut img2_mask = Mat::default();
    imgproc::threshold(
        &img1_gray,
        &mut img1_mask,
        245.0,
        255.0,
        imgproc::THRESH_BINARY,
    )?;
    imgproc::threshold(
        &img2_gray,
        &mut img2_mask,
        245.0,
        255.0,
        imgproc::THRESH_BINARY,
    )?;

    // 对Mask膨胀处理，增加Mask面积
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(3, 3),
        Point::new(-1, -1),
    )?;

    let mut img1_mask_dilated = Mat::default();
    let mut img2_mask_dilated = Mat::default();
    imgproc::dilate(
        &img1_mask,
        &mut img1_mask_dilated,
        &kernel,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        Scalar::default(),
    )?;
    imgproc::dilate(
        &img2_mask,
        &mut img2_mask_dilated,
        &kernel,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

    // 图像修复
    let mut img1_inpaint = Mat::default();
    let mut img2_inpaint = Mat::default();
    photo::inpaint(
        &img1,
        &img1_mask_dilated,
        &mut img1_inpaint,
        5.0,
        photo::INPAINT_NS,
    )?;
    photo::inpaint(
        &img2,
        &img2_mask_dilated,
        &mut img2_inpaint,
        5.0,
        photo::INPAINT_NS,
    )?;

    // 显示处理结果
    highgui::imshow("img1Mask", &img1_mask_dilated)?;
    highgui::imshow("img1修复后", &img1_inpaint)?;
    highgui::imshow("img2Mask", &img2_mask_dilated)?;
    highgui::imshow("img2修复后", &img2_inpaint)?;

    highgui::wait_key(0)?;

    Ok(())
}

