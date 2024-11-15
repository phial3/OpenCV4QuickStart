use anyhow::{Context, Result, Error};
use opencv::{
    core::{Mat, Vector, Scalar},
    highgui,
    imgproc,
    imgcodecs,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    let img_thr = imgcodecs::imread(&format!("{}threshold.png", BASE_PATH), imgcodecs::IMREAD_GRAYSCALE)?;
    if img.empty() || img_thr.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 灰度化
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut img_b = Mat::default();
    let mut img_b_v = Mat::default();
    let mut gray_b = Mat::default();
    let mut gray_b_v = Mat::default();
    let mut gray_t = Mat::default();
    let mut gray_t_v = Mat::default();
    let mut gray_trunc = Mat::default();

    //彩色图像二值化
    imgproc::threshold(&img, &mut img_b, 125.0, 255.0, imgproc::THRESH_BINARY)?;
    imgproc::threshold(&img, &mut img_b_v, 125.0, 255.0, imgproc::THRESH_BINARY_INV)?;
    highgui::imshow("img_B", &img_b)?;
    highgui::imshow("img_B_V", &img_b_v)?;

    //灰度图BINARY二值化
    imgproc::threshold(&gray, &mut gray_b, 125.0, 255.0, imgproc::THRESH_BINARY)?;
    imgproc::threshold(&gray, &mut gray_b_v, 125.0, 255.0, imgproc::THRESH_BINARY_INV)?;
    highgui::imshow("gray_B", &gray_b)?;
    highgui::imshow("gray_B_V", &gray_b_v)?;

    //灰度图像 TOZERO 变换
    imgproc::threshold(&gray, &mut gray_t, 125.0, 255.0, imgproc::THRESH_TOZERO)?;
    imgproc::threshold(&gray, &mut gray_t_v, 125.0, 255.0, imgproc::THRESH_TOZERO_INV)?;
    highgui::imshow("gray_T", &gray_t)?;
    highgui::imshow("gray_T_V", &gray_t_v)?;

    //灰度图像TRUNC变换
    imgproc::threshold(&gray, &mut gray_trunc, 125.0, 255.0, imgproc::THRESH_TRUNC)?;
    highgui::imshow("gray_TRUNC", &gray_trunc)?;

    //灰度图像大津法和三角形法二值化
    let mut img_thr_O = Mat::default();
    let mut img_thr_t = Mat::default();
    imgproc::threshold(&img_thr, &mut img_thr_O, 100.0, 255.0, imgproc::THRESH_BINARY | imgproc::THRESH_OTSU)?;
    imgproc::threshold(&img_thr, &mut img_thr_t, 125.0, 255.0, imgproc::THRESH_BINARY | imgproc::THRESH_TRIANGLE)?;
    highgui::imshow("img_Thr", &img_thr)?;
    highgui::imshow("img_Thr_O", &img_thr_O)?;
    highgui::imshow("img_Thr_T", &img_thr_t)?;

    //灰度图像自适应二值化
    let mut adaptive_mean = Mat::default();
    let mut adaptive_gauss = Mat::default();
    imgproc::adaptive_threshold(&img_thr, &mut adaptive_mean, 255.0, imgproc::ADAPTIVE_THRESH_MEAN_C, imgproc::THRESH_BINARY, 55, 0.0)?;
    imgproc::adaptive_threshold(&img_thr, &mut adaptive_gauss, 255.0, imgproc::ADAPTIVE_THRESH_GAUSSIAN_C, imgproc::THRESH_BINARY, 55, 0.0)?;

    highgui::imshow("adaptive_mean", &adaptive_mean)?;
    highgui::imshow("adaptive_gauss", &adaptive_gauss)?;

    highgui::wait_key(0)?;

    Ok(())
}