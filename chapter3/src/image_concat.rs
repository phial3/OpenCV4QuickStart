use anyhow::Result;
use opencv::{
    prelude::*,
    core::{Mat, Vector},
    highgui,
    imgcodecs,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {

    // 矩阵数组的横竖连接
    let mat1 = Mat::from_slice_2d(&[[1.0f32, 1.0f32]])?;
    let mat2 = Mat::from_slice_2d(&[[2.0f32, 2.0f32]])?;
    let mut vout = Mat::default();
    let mut hout = Mat::default();

    let mut vec_mat: Vector<Mat> = Vector::new();
    vec_mat.push(mat1);
    vec_mat.push(mat2);

    opencv::core::vconcat(&vec_mat, &mut vout)?;
    println!("图像数组竖向连接：\n{:?}", vout);
    opencv::core::hconcat(&vec_mat, &mut hout)?;
    println!("图像数组横向连接：\n{:?}", hout);

    // 矩阵的横竖拼接
    let a = Mat::from_slice_2d(&[[1.0f32, 7.0], [2.0, 8.0]])?;
    let b = Mat::from_slice_2d(&[[4.0f32, 10.0], [5.0, 11.0]])?;
    let mut vc = Mat::default();
    let mut hc = Mat::default();

    let mut vec_mat2: Vector<Mat> = Vector::new();
    vec_mat2.push(a);
    vec_mat2.push(b);

    opencv::core::vconcat(&vec_mat2, &mut vc)?;
    println!("多个图像竖向连接：\n{:?}", vc);
    opencv::core::hconcat(&vec_mat2, &mut hc)?;
    println!("多个图像横向连接：\n{:?}", hc);

    // 读取4个子图像
    let img00 = imgcodecs::imread(&format!("{}lena00.jpg", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    let img01 = imgcodecs::imread(&format!("{}lena01.jpg", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    let img10 = imgcodecs::imread(&format!("{}lena10.jpg", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    let img11 = imgcodecs::imread(&format!("{}lena11.jpg", BASE_PATH), imgcodecs::IMREAD_COLOR)?;

    if img00.empty() || img01.empty() || img10.empty() || img11.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 显示4个子图像
    highgui::named_window("img00", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("img01", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("img10", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("img11", highgui::WINDOW_AUTOSIZE)?;

    highgui::imshow("img00", &img00)?;
    highgui::imshow("img01", &img01)?;
    highgui::imshow("img10", &img10)?;
    highgui::imshow("img11", &img11)?;

    // 图像连接
    let mut img0 = Mat::default();
    let mut img1 = Mat::default();
    let mut img = Mat::default();

    // 图像横向连接
    let mut vec_mat3: Vector<Mat> = Vector::new();
    vec_mat3.push(img00);
    vec_mat3.push(img01);
    opencv::core::hconcat(&vec_mat3, &mut img0)?;

    let mut vec_mat4: Vector<Mat> = Vector::new();
    vec_mat4.push(img10);
    vec_mat4.push(img11);
    opencv::core::hconcat(&vec_mat4, &mut img1)?;

    // 横向连接结果再进行竖向连接
    let mut vec_mat5: Vector<Mat> = Vector::new();
    vec_mat5.push(img0.clone());
    vec_mat5.push(img1.clone());
    opencv::core::vconcat(&vec_mat5, &mut img)?;

    // 显示连接图像的结果
    highgui::named_window("img0", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("img1", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("img", highgui::WINDOW_AUTOSIZE)?;

    // 显示第一次连接的结果
    highgui::imshow("img0", &img0)?;
    highgui::imshow("img1", &img1)?;
    // 显示链接后的完整图像
    highgui::imshow("img", &img)?;

    highgui::wait_key(0)?;

    Ok(())
}