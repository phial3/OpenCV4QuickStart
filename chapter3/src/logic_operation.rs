use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Rect, Scalar, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let mut img = imgcodecs::imread(&(BASE_PATH.to_string() + "lena.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    // 创建两个黑白图像
    let mut img0 = Mat::zeros(200, 200, opencv::core::CV_8UC1)?.to_mat()?;
    let mut img1 = Mat::zeros(200, 200, opencv::core::CV_8UC1)?.to_mat()?;
    // 创建两个矩形区域
    let rect0 = Rect::new(50, 50, 100, 100);
    let rect1 = Rect::new(100, 100, 100, 100);
    // 设置矩形区域为白色,  将 ROI 区域填充为白色
    let _ = img0.roi(rect0)?.try_clone()?.set_to(&Scalar::all(255.0), &opencv::core::no_array())?;
    let _ = img1.roi(rect1)?.try_clone()?.set_to(&Scalar::all(255.0), &opencv::core::no_array())?;

    // 显示原始图像和两个矩形图像
    highgui::imshow("img0", &img0)?;
    highgui::imshow("img1", &img1)?;

    // 进行逻辑运算
    let mut my_and = Mat::default();
    let mut my_or = Mat::default();
    let mut my_xor = Mat::default();
    let mut my_not = Mat::default();
    let mut img_not = Mat::default();

    opencv::core::bitwise_not(&img0, &mut my_not, &opencv::core::no_array())?;
    opencv::core::bitwise_and(&img0, &img1, &mut my_and, &opencv::core::no_array())?;
    opencv::core::bitwise_or(&img0, &img1, &mut my_or, &opencv::core::no_array())?;
    opencv::core::bitwise_xor(&img0, &img1, &mut my_xor, &opencv::core::no_array())?;
    opencv::core::bitwise_not(&img, &mut img_not, &opencv::core::no_array())?;

    highgui::imshow("myAnd", &my_and)?;
    highgui::imshow("myOr", &my_or)?;
    highgui::imshow("myXor", &my_xor)?;
    highgui::imshow("myNot", &my_not)?;
    highgui::imshow("imgNot", &img_not)?;
    highgui::imshow("img", &img)?;

    highgui::wait_key(0)?;

    Ok(())
}