use anyhow::{Context, Result};
use opencv::{
    core::{Mat, Vector},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter4/";

pub(crate) fn run() -> Result<()> {
    // Load images
    let img = imgcodecs::imread(&(BASE_PATH.to_owned() + "apple.jpg"), imgcodecs::IMREAD_COLOR)?;
    let sub_img = imgcodecs::imread(&(BASE_PATH.to_owned() + "sub_apple.jpg"), imgcodecs::IMREAD_COLOR)?;

    if img.empty() || sub_img.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }
    println!("img 图像尺寸: {:?}", img.size());
    println!("sub_img 图像尺寸: {:?}", sub_img.size());

    highgui::imshow("img", &img)?;
    highgui::imshow("sub_img", &sub_img)?;

    // Convert to HSV space
    let mut img_hsv = Mat::default();
    let mut sub_hsv = Mat::default();
    imgproc::cvt_color(&img, &mut img_hsv, imgproc::COLOR_BGR2HSV, 0)?;
    imgproc::cvt_color(&sub_img, &mut sub_hsv, imgproc::COLOR_BGR2HSV, 0)?;

    // Histogram parameters
    let hist_size = Vector::from_slice(&[32, 32]);
    let h_ranges = Vector::from_slice(&[0.0, 180.0]);
    let s_ranges = Vector::from_slice(&[0.0, 256.0]);
    let ranges = Vector::from_slice(&[h_ranges, s_ranges].concat());
    let channels = Vector::from_slice(&[0, 1]);
    let mask = opencv::core::no_array();

    // Compute the 2D histogram for sub image
    let mut hist = Mat::default();
    let mut images: Vector<Mat> = Vector::new();
    images.push(sub_hsv.clone());

    imgproc::calc_hist(
        &images,
        &channels,
        &mask,
        &mut hist,
        &hist_size,
        &ranges,
        false,
    ).context("计算子图像直方图失败").unwrap();

    draw_hist(&hist, opencv::core::NORM_INF, "hist");

    // Compute back projection
    let mut backproj = Mat::default();
    imgproc::calc_back_project(
        &img_hsv,
        &channels,
        &hist,
        &mut backproj,
        &ranges,
        1.0,
    ).context("计算反向投影失败").unwrap();

    highgui::imshow("反向投影后结果", &backproj)?;

    highgui::wait_key(0)?;

    Ok(())
}

fn draw_hist(hist: &Mat, norm_type: i32, name: &str) {
    let hist_w = 512;
    let hist_h = 400;
    let width = 2;
    let mut hist_image = Mat::zeros(hist_h, hist_w, opencv::core::CV_8UC3).unwrap().to_mat().unwrap();
    opencv::core::normalize(&hist, &mut hist_image, 255.0, 0.0, norm_type, -1, &Mat::default()).unwrap();
    highgui::named_window(name, highgui::WINDOW_NORMAL).unwrap();
    highgui::imshow(name, &hist_image).unwrap();
}