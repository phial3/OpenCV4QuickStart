use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Rect, RNG},
    highgui,
    imgcodecs,
    imgproc,
    photo,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter8/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img = imgcodecs::imread(&format!("{}lena.png", BASE_PATH), imgcodecs::IMREAD_COLOR, )?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 转换为灰度图
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // 转换为浮点型
    let mut gray_float = Mat::default();
    gray.convert_to(&mut gray_float, opencv::core::CV_32F, 1.0, 0.0)?;

    // 创建kernel
    let kernel = Mat::from_slice_2d(&[
        &[1.0f32, 1.0, 1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
    ])?;

    // 构建输出图像
    let mut result = Mat::default();
    let rwidth = (gray_float.rows() - kernel.rows()).abs() + 1;
    let rheight = (gray_float.cols() - kernel.cols()).abs() + 1;
    unsafe { result.create_rows_cols(rwidth, rheight, gray_float.typ())? }

    // 计算最优离散傅里叶变换尺寸
    let width = opencv::core::get_optimal_dft_size(gray_float.cols() + kernel.cols() - 1)?;
    let height = opencv::core::get_optimal_dft_size(gray_float.rows() + kernel.rows() - 1)?;

    // 改变输入图像尺寸
    let mut temp_a = Mat::default();
    let a_b = width - gray_float.rows();
    let a_r = height - gray_float.cols();
    opencv::core::copy_make_border(
        &gray_float,
        &mut temp_a,
        0,
        a_b,
        0,
        a_r,
        opencv::core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

    // 改变滤波器尺寸
    let mut temp_b = Mat::default();
    let b_b = width - kernel.rows();
    let b_r = height - kernel.cols();
    opencv::core::copy_make_border(
        &kernel,
        &mut temp_b,
        0,
        b_b,
        0,
        b_r,
        opencv::core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

    // 分别进行离散傅里叶变换
    opencv::core::dft(
        &temp_a.clone(),
        &mut temp_a,
        0,
        gray_float.rows(),
    )?;
    opencv::core::dft(
        &temp_b.clone(),
        &mut temp_b,
        0,
        kernel.rows(),
    )?;

    // 多个傅里叶变换的结果相乘
    let mut temp_result = Mat::default();
    opencv::core::mul_spectrums(
        &temp_a,
        &temp_b,
        &mut temp_result,
        opencv::core::DFT_COMPLEX_OUTPUT,
        false,
    )?;

    // 相乘结果进行逆变换
    // opencv::core::dft(
    //     &temp_result.clone(),
    //     &mut temp_result,
    //     opencv::core::DFT_INVERSE | opencv::core::DFT_SCALE,
    //     result.rows(),
    // )?;
    opencv::core::idft(
        &temp_result.clone(),
        &mut temp_result,
        opencv::core::DFT_SCALE,
        result.rows(),
    )?;

    // 对逆变换结果进行归一化
    opencv::core::normalize(
        &temp_result.clone(),
        &mut temp_result,
        0.0,
        1.0,
        opencv::core::NORM_MINMAX,
        -1,
        &Mat::default(),
    )?;

    // 截取部分结果作为滤波结果
    let roi = Rect::new(0, 0, result.cols(), result.rows());
    temp_result.roi(roi)?.copy_to(&mut result)?;

    // 显示结果
    highgui::imshow("原图像", &gray)?;
    highgui::imshow("滤波结果", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}
