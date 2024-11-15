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
    // 读取图像
    let equal_lena = imgcodecs::imread(&(BASE_PATH.to_owned() + "equalLena.png"), imgcodecs::IMREAD_ANYDEPTH)?;
    if equal_lena.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 定义一个 5x5 的数据矩阵, 验证方框滤波算法的数据矩阵
    let points: [f32; 25] = [
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0,
    ];

    // 创建 Mat 对象来存储数据矩阵
    let data = unsafe { Mat::new_rows_cols_with_data_unsafe_def(5, 5, opencv::core::CV_32FC1, points.as_ptr() as *mut _)? };

    // 将原图像转换为 CV_32F 类型
    let mut equal_lena_32f = Mat::default();
    equal_lena.convert_to(&mut equal_lena_32f, opencv::core::CV_32F, 1.0 / 255.0, 0.0)?;

    // 创建用于存储结果的 Mat 对象
    let mut result_norm = Mat::default();
    let mut result = Mat::default();
    let mut data_sqr_norm = Mat::default();
    let mut data_sqr = Mat::default();
    let mut equal_lena_32f_sqr = Mat::default();

    // 进行 boxFilter 和 sqrBoxFilter 滤波, border_type=4, BorderTypes::BORDER_DEFAULT
    imgproc::box_filter(&equal_lena, &mut result_norm, -1, Size::new(3, 3), Point::new(-1, -1), true, 4)?;  //进行归一化
    imgproc::box_filter(&equal_lena, &mut result, -1, Size::new(3, 3), Point::new(-1, -1), false, 4)?;      //不进行归一化

    imgproc::sqr_box_filter(&data, &mut data_sqr_norm, -1, Size::new(3, 3), Point::new(-1, -1), true, 4)?;  //进行归一化
    imgproc::sqr_box_filter(&data, &mut data_sqr, -1, Size::new(3, 3), Point::new(-1, -1), false, 4)?;      //不进行归一化

    imgproc::sqr_box_filter(&equal_lena_32f, &mut equal_lena_32f_sqr, -1, Size::new(3, 3), Point::new(-1, -1), true, 4)?;

    // 显示处理结果
    highgui::imshow("resultNorm", &result_norm)?;
    highgui::imshow("result", &result)?;
    highgui::imshow("equalLena_32FSqr", &equal_lena_32f_sqr)?;

    // 等待按键事件
    highgui::wait_key(0)?;

    Ok(())
}