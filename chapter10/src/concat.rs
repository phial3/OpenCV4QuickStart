use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Point2f, Point3f, Scalar, Size, Vector, TermCriteria},
    imgcodecs,
    imgproc,
    highgui,
    calib3d,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 创建矩阵数组并进行横竖连接
    let mut mat_array = Vector::<Mat>::new();
    mat_array.push(Mat::new_rows_cols_with_default(1, 2, opencv::core::CV_32FC1, Scalar::all(1.0))?);
    mat_array.push(Mat::new_rows_cols_with_default(1, 2, opencv::core::CV_32FC1, Scalar::all(2.0))?);

    // 竖向连接
    let mut vout = Mat::default();
    opencv::core::vconcat(&mat_array, &mut vout)?;
    println!("图像数组竖向连接：{:?}", vout);

    // 横向连接
    let mut hout = Mat::default();
    opencv::core::hconcat(&mat_array, &mut hout)?;
    println!("图像数组横向连接：{:?}", hout);

    // 矩阵 A 和 B 进行横竖拼接
    let a = Mat::from_slice_2d(&[
        &[1.0, 7.0],
        &[2.0, 8.0],
    ])?;
    let b = Mat::from_slice_2d(&[
        &[4.0, 10.0],
        &[5.0, 11.0],
    ])?;

    let mut arr = Vector::<Mat>::new();
    arr.push(a);
    arr.push(b);
    // 竖向拼接
    let mut vC = Mat::default();
    opencv::core::vconcat(&arr, &mut vC)?;
    println!("多个图像竖向连接：{:?}", vC);

    // 横向拼接
    let mut hC = Mat::default();
    opencv::core::hconcat(&arr, &mut hC)?;
    println!("多个图像横向连接：{:?}", hC);

    // 读取 4 个子图像
    let img00 = imgcodecs::imread(&format!("{}lena00.jpg", BASE_PATH), imgcodecs::IMREAD_UNCHANGED)?;
    let img01 = imgcodecs::imread(&format!("{}lena01.jpg", BASE_PATH), imgcodecs::IMREAD_UNCHANGED)?;
    let img10 = imgcodecs::imread(&format!("{}lena10.jpg", BASE_PATH), imgcodecs::IMREAD_UNCHANGED)?;
    let img11 = imgcodecs::imread(&format!("{}lena11.jpg", BASE_PATH), imgcodecs::IMREAD_UNCHANGED)?;
    if img00.empty() || img01.empty() || img10.empty() || img11.empty() {
        panic!("读取图像错误，请确认图像文件是否正确")
    }

    // 显示 4 个子图像
    highgui::imshow("img00", &img00)?;
    highgui::imshow("img01", &img01)?;
    highgui::imshow("img10", &img10)?;
    highgui::imshow("img11", &img11)?;

    // 图像连接
    let mut img0 = Mat::default();
    let mut img1 = Mat::default();
    let mut img = Mat::default();

    let mut arr_00 = Vector::<Mat>::new();
    arr_00.push(img00);
    arr_00.push(img01);
    let mut arr_10 = Vector::<Mat>::new();
    arr_10.push(img10);
    arr_10.push(img11);

    // 横向连接
    opencv::core::hconcat(&arr_00, &mut img0)?;
    opencv::core::hconcat(&arr_10, &mut img1)?;

    let mut arr = Vector::<Mat>::new();
    arr.push(img0.clone());
    arr.push(img1.clone());
    // 横向连接的结果再竖向连接
    opencv::core::vconcat(&arr, &mut img)?;

    // 显示连接图像的结果
    highgui::imshow("img0", &img0)?;
    highgui::imshow("img1", &img1)?;
    highgui::imshow("img", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}
