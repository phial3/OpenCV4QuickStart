use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Rect, Range, Point, Size, Scalar, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let mut img = imgcodecs::imread(&(BASE_PATH.to_string() + "lena.png"), imgcodecs::IMREAD_COLOR)?;
    let noobcv = imgcodecs::imread(&(BASE_PATH.to_string() + "noobcv.jpg"), imgcodecs::IMREAD_COLOR)?;

    if img.empty() || noobcv.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    // 定义图像变量
    let mut ROI1 = Mat::default();
    let mut ROI2 = Mat::default();
    let mut ROI2_copy = Mat::default();
    let mut mask = Mat::default();
    let mut img2 = Mat::default();
    let mut img_copy = Mat::default();
    let mut img_copy2 = Mat::default();

    // 调整图像大小
    imgproc::resize(&noobcv, &mut mask, Size { width: 200, height: 200 }, 0.0, 0.0, imgproc::INTER_LINEAR)?;

    // 浅拷贝
    img2 = img.clone();

    // 深拷贝
    img.copy_to(&mut img_copy)?;
    img.copy_to(&mut img_copy2)?;

    // 两种方式截取ROI区域
    let rect = Rect::new(206, 206, 200, 200);  // 定义ROI区域
    //截图
    ROI1 = img.roi(rect)?.try_clone()?;
    //第二种截图方式
    let mut ranges = Vector::new();
    ranges.push(Range::new(300, 500)?);
    ranges.push(Range::new(300, 500)?);
    ROI2 = img.ranges(&ranges)?.try_clone()?;  // 使用Range截取

    // 深拷贝ROI区域
    img.ranges(&ranges)?.copy_to(&mut ROI2_copy)?;

    // 将mask图像拷贝到ROI1, 在图像中加入部分图像
    mask.copy_to(&mut ROI1)?;

    // 显示不同的图像
    highgui::imshow("加入noobcv后图像", &img)?;
    highgui::imshow("ROI对ROI2的影响", &ROI2)?;
    highgui::imshow("深拷贝的ROI2_copy", &ROI2_copy)?;

    // 绘制一个圆形
    imgproc::circle(&mut img,
                    Point::new(300, 300),
                    20,
                    Scalar::new(0.0, 0.0, 255.0, 0.0),
                    -1,
                    imgproc::LINE_8,
                    0
    )?;

    // 显示图像
    highgui::imshow("浅拷贝的img2", &img2)?;
    highgui::imshow("深拷贝的img_copy", &img_copy)?;
    highgui::imshow("深拷贝的img_copy2", &img_copy2)?;
    highgui::imshow("画圆对ROI1的影响", &ROI1)?;

    highgui::wait_key(0)?;

    Ok(())
}