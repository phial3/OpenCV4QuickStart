use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // LUT查找表第一层
    let mut lut_first = [0u8; 256];
    for i in 0..256 {
        lut_first[i] = match i {
            0..=100 => 0,
            101..=200 => 100,
            _ => 255,
        };
    }
    let lut_one = Mat::from_slice(&lut_first)?.try_clone()?;

    // LUT查找表第二层
    let mut lut_second = [0u8; 256];
    for i in 0..256 {
        lut_second[i] = match i {
            0..=100 => 0,
            101..=150 => 100,
            151..=200 => 150,
            _ => 255,
        };
    }
    let lut_two = Mat::from_slice(&lut_second)?.try_clone()?;

    // LUT查找表第三层
    let mut lut_third = [0u8; 256];
    for i in 0..256 {
        lut_third[i] = match i {
            0..=100 => 100,
            101..=200 => 200,
            _ => 255,
        };
    }
    let lut_three = Mat::from_slice(&lut_third)?.try_clone()?;

    // 拥有三通道的LUT查找表矩阵
    let mut merge_mats: Vector<Mat> = Vector::new();
    merge_mats.push(lut_one.clone());
    merge_mats.push(lut_two);
    merge_mats.push(lut_three);
    let mut lut_tree = Mat::default();
    opencv::core::merge(&merge_mats, &mut lut_tree)?;

    // 计算图像的查找表
    let img = imgcodecs::imread(format!("{}lena.png", BASE_PATH).as_str(), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut out0 =Mat::default();
    let mut out1 = Mat::default();
    let mut out2 = Mat::default();

    // opencv::core::lut 函数（在C++中称为 cv::LUT）用于对图像应用查找表（Look-Up Table, LUT）。
    // 查找表是一种预先计算好的映射关系，可以将输入图像中的每个像素值映射到一个新的值。这种技术常用于图像的对比度调整、颜色变换等操作。
    opencv::core::lut(&gray, &lut_one, &mut out0)?;
    opencv::core::lut(&img, &lut_one, &mut out1)?;
    opencv::core::lut(&img, &lut_tree, &mut out2)?;

    highgui::imshow("out0", &out0)?;
    highgui::imshow("out1", &out1)?;
    highgui::imshow("out2", &out2)?;

    highgui::wait_key(0)?;

    Ok(())
}