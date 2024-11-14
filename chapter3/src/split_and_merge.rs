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
    if img.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    // 转换为HSV
    let mut hsv = Mat::default();
    imgproc::cvt_color(&img, &mut hsv, imgproc::COLOR_RGB2HSV, 0)?;

    // 用于存放结果的Mat
    let mut imgs0 = Mat::default();
    let mut imgs1 = Mat::default();
    let mut imgs2 = Mat::default();
    let mut imgv0 = Mat::default();
    let mut imgv1 = Mat::default();
    let mut imgv2 = Mat::default();
    let mut result0 = Mat::default();
    let mut result1 = Mat::default();
    let mut result2 = Mat::default();

    // 输入数组参数的多通道分离与合并
    let mut channels = Vector::<Mat>::new();
    opencv::core::split(&img, &mut channels)?;

    // 获取各个通道
    imgs0 = channels.get(0)?.clone();
    imgs1 = channels.get(1)?.clone();
    imgs2 = channels.get(2)?.clone();

    // 显示RGB各通道
    highgui::imshow("RGB-B通道", &imgs0)?;
    highgui::imshow("RGB-G通道", &imgs1)?;
    highgui::imshow("RGB-R通道", &imgs2)?;

    // 将通道数变得不统一
    channels.set(2, img.clone())?;
    opencv::core::merge(&channels, &mut result0)?;

    // 创建全零矩阵
    let zero = Mat::zeros(img.rows(), img.cols(), opencv::core::CV_8UC1)?.to_mat()?;

    // 重置channels用于合并
    let mut merge_channels = Vector::<Mat>::new();
    merge_channels.push(zero.clone());
    merge_channels.push(channels.get(1)?.clone());
    merge_channels.push(zero.clone());
    opencv::core::merge(&merge_channels, &mut result1)?;
    highgui::imshow("result1", &result1)?;

    // 输入vector参数的多通道分离与合并
    let mut hsv_channels = Vector::<Mat>::new();
    opencv::core::split(&hsv, &mut hsv_channels)?;

    // 获取HSV各通道
    imgv0 = hsv_channels.get(0)?.clone();
    imgv1 = hsv_channels.get(1)?.clone();
    imgv2 = hsv_channels.get(2)?.clone();

    // 显示HSV各通道
    highgui::imshow("HSV-H通道", &imgv0)?;
    highgui::imshow("HSV-S通道", &imgv1)?;
    highgui::imshow("HSV-V通道", &imgv2)?;

    // 将通道数变得不统一
    hsv_channels.push(hsv.clone());
    opencv::core::merge(&hsv_channels, &mut result2)?;

    highgui::wait_key(0)?;

    Ok(())
}