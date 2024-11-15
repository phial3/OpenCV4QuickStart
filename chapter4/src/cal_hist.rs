use anyhow::{Context, Result};
use opencv::{
    core::{Mat, Rect, Scalar, Vector},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter4/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&(BASE_PATH.to_string() + "apple.jpg"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }
    println!("图像加载成功，图像尺寸：{:?}", img.size());

    // 转换为灰度图
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    println!("灰度图尺寸: {:?}", gray.size());

    let mut images: Vector<Mat> = Vector::new();
    images.push(gray.clone());
    let channels = Vector::from_slice(&[0]);
    let mask_roi = opencv::core::no_array();
    let hist_size = Vector::from_slice(&[256]);
    let ranges = Vector::from_slice(&[0.0, 255.0]);
    let mut hist = Mat::default();
    imgproc::calc_hist(
        &images,
        &channels,
        &mask_roi,
        &mut hist,
        &hist_size,
        &ranges,
        false,
    ).context("计算直方图失败").unwrap();

    // 准备绘制直方图
    let hist_w = 512;
    let hist_h = 400;
    let width = 2;
    let mut hist_image = Mat::zeros(hist_w, hist_h, opencv::core::CV_8UC3)?.to_mat()?;

    // 绘制直方图
    for i in 1..=hist.rows() {
        let x1 = width * (i - 1);
        let x2 = width * i;
        let y1 = hist_h - 1;
        let y2 = hist_h - (hist.at_2d::<f32>(i - 1, 0)? / 15.0).round() as i32;

        // 使用 Rect 类型进行绘制
        let rect = Rect::new(x1, y2, x2 - x1, y1 - y2);
        imgproc::rectangle(
            &mut hist_image,
            rect,
            Scalar::all(255.0),
            -1,  // 填充矩形
            imgproc::LINE_8,
            0,
        )?;
    }

    // 显示图像
    highgui::named_window("histImage", highgui::WINDOW_AUTOSIZE)?;
    highgui::imshow("histImage", &hist_image)?;
    highgui::imshow("gray", &gray)?;

    highgui::wait_key(0)?;

    println!("直方图计算完成！");

    Ok(())
}