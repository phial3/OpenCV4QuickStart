use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat},
    prelude::*,
};

pub(crate) fn run() -> Result<()> {
    // 初始化数据
    let a: [f32; 12] = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0];

    // 创建单通道矩阵（3行4列）
    let img = unsafe { Mat::new_rows_cols_with_data_unsafe_def(3, 4, opencv::core::CV_32FC1, a.as_ptr() as *mut _)? };

    // 创建多通道矩阵（2行3列，2通道）
    let imgs = unsafe { Mat::new_rows_cols_with_data_unsafe_def(2, 3, opencv::core::CV_32FC2, a.as_ptr() as *mut _)? };

    // 计算imgs的均值
    let my_mean = opencv::core::mean(&imgs, &opencv::core::no_array())?;
    println!("imgs 均值={:?}", my_mean);
    println!("imgs 第一个通道的均值={}    imgs第二个通道的均值={}", my_mean[0], my_mean[1]);
    println!();

    // 计算img的均值和标准差
    let mut my_mean_mat = Mat::default();
    let mut my_stddev_mat = Mat::default();
    opencv::core::mean_std_dev(&img, &mut my_mean_mat, &mut my_stddev_mat, &opencv::core::no_array())?;
    println!("img 均值={:?}    ", my_mean_mat);
    println!("img 标准差={:?}", my_stddev_mat);
    println!();

    // 计算imgs的均值和标准差
    opencv::core::mean_std_dev(&imgs, &mut my_mean_mat, &mut my_stddev_mat, &opencv::core::no_array())?;
    println!("imgs 均值={:?}    ", my_mean_mat);
    println!("imgs 标准差={:?}", my_stddev_mat);

    Ok(())
}