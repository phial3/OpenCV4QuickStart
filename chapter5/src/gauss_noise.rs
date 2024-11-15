use anyhow::Result;
use opencv::{
    prelude::*,
    core::{Mat, Size, Point, RNG, BorderTypes},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter5/";

pub(crate) fn run() -> Result<()> {
    let lena = imgcodecs::imread(&format!("{}{}", BASE_PATH, "lena.png"), imgcodecs::IMREAD_COLOR)?;
    let equal_lena = imgcodecs::imread(&format!("{}{}", BASE_PATH, "equalLena.png"), imgcodecs::IMREAD_ANYDEPTH)?;
    if lena.empty() || equal_lena.empty() {
        println!("请确认图像文件名称是否正确");
        return Ok(());
    }

    // 生成与原图像同尺寸、数据类型和通道数的矩阵
    let height = lena.rows();
    let width = lena.cols();
    let lena_type = lena.typ();

    let mut lena_noise = Mat::zeros(height, width, lena_type)?.to_mat()?;
    let mut equal_lena_noise = Mat::zeros(height, width, equal_lena.typ())?.to_mat()?;

    highgui::imshow("lena原图", &lena)?;
    highgui::imshow("equalLena原图", &equal_lena)?;

    // 创建均值和标准差的输入数组
    let lena_mean = unsafe { Mat::new_rows_cols_with_data_unsafe_def(1, 1, opencv::core::CV_64F, [10.0].as_ptr() as *mut _)? };
    let lena_stddev = unsafe { Mat::new_rows_cols_with_data_unsafe_def(1, 1, opencv::core::CV_64F, [20.0].as_ptr() as *mut _)? };

    let equal_lena_mean = unsafe { Mat::new_rows_cols_with_data_unsafe_def(1, 1, opencv::core::CV_64F, [15.0].as_ptr() as *mut _)? };
    let equal_lena_stddev = unsafe { Mat::new_rows_cols_with_data_unsafe_def(1, 1, opencv::core::CV_64F, [30.0].as_ptr() as *mut _)? };

    // 随机数生成
    let mut rng = RNG::default()?;
    // 生成高斯分布随机数
    rng.fill(&mut lena_noise, opencv::core::RNG_NORMAL, &lena_mean, &lena_stddev, false)?;
    rng.fill(&mut equal_lena_noise, opencv::core::RNG_NORMAL, &equal_lena_mean, &equal_lena_stddev, false)?;

    highgui::imshow("三通道高斯噪声", &lena_noise)?;
    highgui::imshow("单通道高斯噪声", &equal_lena_noise)?;

    // 在图像中添加高斯噪声
    let mut lena_with_noise = lena.clone();
    let mut equal_lena_with_noise = equal_lena.clone();

    opencv::core::add(&lena, &lena_noise, &mut lena_with_noise, &Mat::default(), -1)?;
    opencv::core::add(&equal_lena, &equal_lena_noise, &mut equal_lena_with_noise, &Mat::default(), -1)?;

    // 显示添加高斯噪声后的图像
    highgui::imshow("lena添加噪声", &lena_with_noise)?;
    highgui::imshow("equalLena添加噪声", &equal_lena_with_noise)?;

    highgui::wait_key(0)?;

    Ok(())
}