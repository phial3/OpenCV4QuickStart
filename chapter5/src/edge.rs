use anyhow::Result;
use opencv::{
    prelude::*,
    core::{Mat, Size, Point, BorderTypes},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {

    //创建边缘检测滤波器
    let kernel1 = Mat::from_slice_2d(&[[1.0, -1.0]])?; //X方向边缘检测滤波器
    let kernel2 = Mat::from_slice_2d(&[[1.0, 0.0, -1.0]])?; //X方向边缘检测滤波器
    let kernel3 = Mat::from_slice_2d(&[[1.0], [0.0], [-1.0]])?; //Y方向边缘检测滤波器
    let kernel_xy = Mat::from_slice_2d(&[[1.0, 0.0], [0.0, -1.0]])?; //由左上到右下方向边缘检测滤波器
    let kernel_yx = Mat::from_slice_2d(&[[0.0, -1.0], [1.0, 0.0]])?; //由右上到左下方向边缘检测滤波器

    //读取图像，黑白图像边缘检测结果较为明显
    let img = imgcodecs::imread(&(BASE_PATH.to_string() + "equalLena.png"), imgcodecs::IMREAD_COLOR).unwrap();
    if img.empty() {
        panic!("Please confirm the image file name is correct.");
    }

    let mut result1 = Mat::default();
    let mut result2 = Mat::default();
    let mut result3 = Mat::default();
    let mut result4 = Mat::default();
    let mut result5 = Mat::default();
    let mut result6 = Mat::default();

    //以[1 -1]检测水平方向边缘
    imgproc::filter_2d(&img, &mut result1, opencv::core::CV_16S, &kernel1, Point::new(0, 0), 0.0, opencv::core::BORDER_DEFAULT).unwrap();
    opencv::core::convert_scale_abs(&result1.clone(), &mut result1, 1.0, 0.0).unwrap();

    //以[1 0 -1]检测水平方向边缘
    imgproc::filter_2d(&img, &mut result2, opencv::core::CV_16S, &kernel2, Point::new(0, 0), 0.0, opencv::core::BORDER_DEFAULT).unwrap();
    opencv::core::convert_scale_abs(&result2.clone(), &mut result2, 1.0, 0.0).unwrap();

    //以[1 0 -1]'检测由垂直方向边缘
    imgproc::filter_2d(&img, &mut result3, opencv::core::CV_16S, &kernel3, Point::new(0, 0), 0.0, opencv::core::BORDER_DEFAULT).unwrap();
    opencv::core::convert_scale_abs(&result3.clone(), &mut result3, 1.0, 0.0).unwrap();

    //整幅图像的边缘
    opencv::core::add(&result2, &result3, &mut result6, &opencv::core::no_array(), -1).unwrap();
    //检测由左上到右下方向边缘
    imgproc::filter_2d(&img, &mut result4, opencv::core::CV_16S, &kernel_xy, Point::new(0, 0), 0.0, opencv::core::BORDER_DEFAULT).unwrap();
    opencv::core::convert_scale_abs(&result4.clone(), &mut result4, 1.0, 0.0).unwrap();

    //检测由右上到左下方向边缘
    imgproc::filter_2d(&img, &mut result5, opencv::core::CV_16S, &kernel_yx, Point::new(0, 0), 0.0, opencv::core::BORDER_DEFAULT).unwrap();
    opencv::core::convert_scale_abs(&result5.clone(), &mut result5, 1.0, 0.0).unwrap();

    // Show the results
    highgui::imshow("result1", &result1)?;
    highgui::imshow("result2", &result2)?;
    highgui::imshow("result3", &result3)?;
    highgui::imshow("result4", &result4)?;
    highgui::imshow("result5", &result5)?;
    highgui::imshow("result6", &result6)?;

    highgui::wait_key(0)?;

    Ok(())
}